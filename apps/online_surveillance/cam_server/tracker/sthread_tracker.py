import zmq
import random
import logging
import numpy as np
from typing import List

from mot.utils import Timer, Config
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.tracker import Tracker, TRACKER_REGISTRY

from .tracker import LocalTracker


@TRACKER_REGISTRY.register()
class SingleThreadTracker(LocalTracker):
    def __init__(self, detector: Config, encoders: List[Config], matcher: Config, predictor: Config,
                 central_address: str, **kwargs):
        self.detector = build_detector(detector)
        self.encoders = [build_encoder(encoder) for encoder in encoders]
        self.detection_filters = []
        self.identifier = random.randint(0, 1000000)

        self.zmq_context = zmq.Context()
        self.subscribe_socket = self.zmq_context.socket(zmq.SUB)
        self.subscribe_socket.connect('tcp://' + central_address)
        self.subscribe_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        super(SingleThreadTracker, self).__init__(self.detector, self.encoders, matcher, predictor, **kwargs)

    @Timer.timer('all')
    def tick(self, img: np.ndarray):
        # If any global ID data is available, add global information to local tracklets.
        try:
            while True:
                data = self.subscribe_socket.recv_string(flags=zmq.NOBLOCK)
                data = list(map(lambda x: int(float(x)), data.strip().split(' ')))
                if data[0] == self.identifier:
                    logging.getLogger('MOT').info('Received data: {} from central server'.format(data))
                    _, localID, globalID = data
                    for tracklet in self.tracklets_active:
                        if tracklet.id == localID:
                            logging.getLogger('MOT').info(
                                'Local target #{} is identified as global ID {}'.format(localID, globalID))
                            tracklet.globalID = globalID
        except zmq.ZMQError:
            pass

        super(SingleThreadTracker, self).tick(img)

    def terminate(self) -> None:
        super(SingleThreadTracker, self).terminate()
        self.subscribe_socket.close()
        self.zmq_context.destroy()

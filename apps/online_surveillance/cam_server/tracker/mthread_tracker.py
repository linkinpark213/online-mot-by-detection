import zmq
import time
import random
import logging
import traceback
import numpy as np
from typing import List
from threading import Thread, Lock

from mot.utils import Timer, Config
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.tracker import Tracker, TRACKER_REGISTRY, TrackerState

from .tracker import LocalTracker


class DetectorThread(Thread):
    def __init__(self, config: Config, tracker: Tracker, lock: Lock, next_lock: Lock):
        super().__init__()
        self.config = config
        self.tracker = tracker
        self.lock = lock
        self.next_lock = next_lock
        self.running = True

    def run(self) -> None:
        detector = None
        try:
            if self.config is not None:
                detector = build_detector(self.config)
                while self.running:
                    if self.lock.acquire(timeout=1):
                        self.tracker.latest_detections = detector.detect(self.tracker.frame)
                        self.next_lock.release()
        except Exception as e:
            traceback.print_exc()
            self.running = False
            raise e
        finally:
            if hasattr(detector, 'destroy'):
                detector.destroy()
            logging.getLogger('MOT').info('Detector thread terminated.')


class EncoderThread(Thread):
    def __init__(self, config: Config, tracker: Tracker, lock: Lock, next_lock: Lock):
        super().__init__()
        self.config = config
        self.tracker = tracker
        self.lock = lock
        self.next_lock = next_lock
        self.running = True

    def run(self) -> None:
        encoder = build_encoder(self.config)
        while self.running:
            if self.lock.acquire(timeout=1):
                features = encoder(self.tracker.latest_detections, self.tracker.frame)
                for i, feature in enumerate(features):
                    self.tracker.latest_features[i][encoder.name] = feature
                self.next_lock.release()
        logging.getLogger('MOT').info('Encoder thread terminated.')


class GlobalIDListenerThread(Thread):
    def __init__(self, tracker: Tracker, lock: Lock, central_address: str, identifier: int):
        super().__init__()
        self.tracker = tracker
        self.lock = lock
        self.central_address = central_address
        self.identifier = identifier
        self.running = True
        context = zmq.Context()
        self.subscribe_socket = context.socket(zmq.SUB)
        self.subscribe_socket.connect('tcp://' + central_address)
        self.subscribe_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        self.queue = []

    def run(self) -> None:
        while self.running:
            try:
                data = self.subscribe_socket.recv_string(flags=zmq.NOBLOCK)
                data = list(map(lambda x: int(float(x)), data.strip().split(' ')))
                logging.getLogger('MOT').debug('Received data: {} from central server'.format(data))
                if data[0] == self.identifier:
                    # Save to queue for possible later processing (if lock can't be acquired)
                    self.queue.append(data)

                if self.lock.acquire(timeout=1):
                    while len(self.queue) > 0:
                        _, localID, globalID = self.queue.pop(0)
                        for tracklet in self.tracker.tracklets_active:
                            if tracklet.id == localID:
                                logging.getLogger('MOT').info(
                                    'Local target #{} is identified as global ID {}'.format(localID, globalID))
                                tracklet.globalID = globalID
                    self.lock.release()
            except zmq.ZMQError:
                time.sleep(0.1)
            except Exception as e:
                self.subscribe_socket.close()
                self.running = False
                raise e

        logging.getLogger('MOT').info('ID listener thread terminated.')


@TRACKER_REGISTRY.register()
class MultiThreadTracker(LocalTracker):
    def __init__(self, detector: Config, encoders: List[Config], matcher: Config, predictor: Config,
                 central_address: str, **kwargs):
        self.detect_lock = Lock()
        self.filter_lock = Lock()
        self.encode_locks = [Lock() for _ in encoders]
        self.tracklets_lock = Lock()
        self.after_locks = [Lock() for _ in encoders]
        self.detect_lock.acquire()
        self.filter_lock.acquire()
        for lock in self.encode_locks:
            lock.acquire()
        for lock in self.after_locks:
            lock.acquire()

        self.identifier = random.randint(0, 1000000)

        self.detector_thread = DetectorThread(detector, self, self.detect_lock, self.filter_lock)
        self.encoder_threads = [EncoderThread(encoders[i], self, self.encode_locks[i], self.after_locks[i]) for i in
                                range(len(encoders))]
        self.listener_thread = GlobalIDListenerThread(self, self.tracklets_lock, central_address, self.identifier)

        self.state: TrackerState = TrackerState()
        self.latest_features = []

        super(MultiThreadTracker, self).__init__(None, None, matcher, predictor, **kwargs)

        self.detector_thread.start()
        for thread in self.encoder_threads:
            thread.start()
        self.listener_thread.start()

    @Timer.timer('all')
    def tick(self, img: np.ndarray):
        """
        Detect, encode and match, following the tracking-by-detection paradigm.
        The tracker works online. For each new frame, the tracker ticks once.

        Args:
            img: A 3D numpy array with shape (H, W, 3). The new frame in the sequence.
        """
        self.frame_num += 1
        self.frame = img.copy()
        self.timestamp = time.time()

        # Prediction
        self.predict(img)

        # Detection and encoding
        self.detect_lock.release()

        # Filtering
        self.filter_lock.acquire()
        for filter in self.detection_filters:
            self.latest_detections = filter(self.latest_detections)
        self.latest_features = [{} for i in range(len(self.latest_detections))]

        for lock in self.encode_locks:
            lock.release()

        # Wait for detection and encoding to finish
        for lock in self.after_locks:
            lock.acquire()

        self.tracklets_lock.acquire()
        # Data Association
        row_ind, col_ind = self.match(self.tracklets_active, self.latest_features)

        # Tracklet Update
        self.update(row_ind, col_ind, self.latest_detections, self.latest_features)

        # Log status
        self.log(frame_num=self.frame_num,
                 dets=len(self.latest_detections),
                 matches=len(row_ind),
                 targets=len(self.tracklets_active))
        self.tracklets_lock.release()

    def terminate(self) -> None:
        super().terminate()
        self.logger.info('Setting all threads to "not running"')
        self._stop_all()

    def _stop_all(self) -> None:
        self.detector_thread.running = False
        for thread in self.encoder_threads:
            thread.running = False
        self.listener_thread.running = False

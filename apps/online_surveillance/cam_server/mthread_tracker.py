import time
import traceback
import numpy as np
from typing import List
import pycuda.driver as cuda
from threading import Thread, Lock

from mot.utils import Timer, Config
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.associate import build_matcher
from mot.predict import build_predictor
from mot.filter import build_detection_filter
from mot.tracker import Tracker, TRACKER_REGISTRY


class DetectorThread(Thread):
    def __init__(self, config: Config, tracker: Tracker, lock: Lock, next_lock: Lock):
        super().__init__()
        self.config = config
        self.tracker = tracker
        self.lock = lock
        self.next_lock = next_lock
        self.running = True

    def run(self) -> None:
        cuda.init()
        device = cuda.Device(0)  # enter your Gpu id here
        ctx = device.make_context()

        try:
            if self.config is not None:
                detector = build_detector(self.config)
                while self.running:
                    if self.lock.acquire(timeout=500):
                        self.tracker.latest_detections = detector.detect(self.tracker.frame)
                        self.next_lock.release()
        except Exception as e:
            traceback.print_exc()
        finally:
            ctx.pop()


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
            if self.lock.acquire(timeout=500):
                features = encoder(self.tracker.latest_detections, self.tracker.frame)
                for i, feature in enumerate(features):
                    self.tracker.latest_features[i][encoder.name] = feature
                self.next_lock.release()


@TRACKER_REGISTRY.register()
class MultiThreadTracker(Tracker):
    def __init__(self, detector: Config, matcher: Config, encoders: List[Config], predictor: Config, **kwargs):
        self.detect_lock = Lock()
        self.filter_lock = Lock()
        self.encode_lock = Lock()
        self.after_locks = [Lock() for _ in encoders]
        self.detect_lock.acquire()
        self.filter_lock.acquire()
        self.encode_lock.acquire()
        for lock in self.after_locks:
            lock.acquire()

        self.detector_thread = DetectorThread(detector, self, self.detect_lock, self.filter_lock)

        self.encoder_threads = [EncoderThread(encoders[i], self, self.encode_lock, self.after_locks[i]) for i in
                                range(len(encoders))]

        self.latest_detections = []
        self.latest_features = []

        matcher = build_matcher(matcher)
        predictor = build_predictor(predictor)
        if 'detection_filters' in kwargs.keys():
            kwargs['detection_filters'] = [build_detection_filter(filter_cfg) for filter_cfg in
                                           kwargs['detection_filters']]
        super().__init__(None, None, matcher, predictor, **kwargs)

        self.detector_thread.start()
        for thread in self.encoder_threads:
            thread.start()

    @Timer.timer('all')
    def tick(self, img: np.ndarray):
        """
        Detect, encode and match, following the tracking-by-detection paradigm.
        The tracker works online. For each new frame, the tracker ticks once.

        Args:
            img: A 3D numpy array with shape (H, W, 3). The new frame in the sequence.
        """
        self.frame_num += 1
        self.frame = img
        self.timestamp = time.time()

        # Prediction
        self.predict(img)

        # Detection and encoding
        self.detect_lock.release()

        # Filtering
        self.filter_lock.acquire(timeout=1000)
        for filter in self.detection_filters:
            self.latest_detections = filter(self.latest_detections)
        self.latest_features = [{} for i in range(len(self.latest_detections))]
        self.encode_lock.release()

        # Wait for detection and encoding to finish
        for lock in self.after_locks:
            lock.acquire()

        # Data Association
        row_ind, col_ind = self.match(self.tracklets_active, self.latest_features)

        # Tracklet Update
        self.update(row_ind, col_ind, self.latest_detections, self.latest_features)

        # Log status
        self.log(frame_num=self.frame_num,
                 dets=len(self.latest_detections),
                 matches=len(row_ind),
                 targets=len(self.tracklets_active))

    def terminate(self) -> None:
        super().terminate()
        self.detector_thread.running = False
        for thread in self.encoder_threads:
            thread.running = False

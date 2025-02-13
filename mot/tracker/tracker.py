import time
import logging
import numpy as np
from typing import List, Dict, Union, Tuple

from mot.encode import Encoder
from mot.detect import Detector
from mot.associate import Matcher
from mot.predict import Predictor
from mot.structures import Tracklet
from mot.utils import Registry, Timer
from mot.filter import DetectionFilter
from mot.structures import Detection, Prediction

__all__ = ['Tracker', 'TrackerState', 'TRACKER_REGISTRY', 'build_tracker']

TRACKER_REGISTRY = Registry('trackers')


class TrackerState:
    def __init__(self):
        self.max_id: int = 0
        self.tracklets_active: List[Tracklet] = []
        self.tracklets_finished: List[Tracklet] = []
        self.latest_detections: List[Detection] = []
        self.frame_num: int = 0
        self.frame: Union[None, np.ndarray] = None
        self.timestamp: float = 0


class Tracker:
    def __init__(self, detector: Detector,
                 encoders: List[Encoder],
                 matcher: Matcher,
                 predictor: Predictor = None,
                 detection_filters: List[DetectionFilter] = None,
                 max_ttl: int = 30,
                 max_feature_history: int = 30,
                 max_detection_history: int = 3000,
                 min_time_lived: int = 0,
                 keep_finished_tracks: bool = False,
                 **kwargs) -> None:
        self.detector: Detector = detector
        self.encoders: List[Encoder] = encoders
        self.matcher: Matcher = matcher
        self.predictor: Predictor = predictor
        self.detection_filters = detection_filters
        self.max_ttl: int = max_ttl
        self.max_feature_history: int = max_feature_history
        self.max_detection_history: int = max_detection_history
        self.min_time_lived: int = min_time_lived
        self.keep_finished_tracks: bool = keep_finished_tracks
        self.logger: logging.Logger = logging.getLogger('MOT')
        self.state: TrackerState = TrackerState()

    def clear(self) -> None:
        self.max_id = 1
        self.tracklets_active = []
        self.tracklets_finished = []
        self.frame_num = 0
        self.latest_detections = []

    @property
    def tracklets_active(self):
        return self.state.tracklets_active

    @tracklets_active.setter
    def tracklets_active(self, value):
        self.state.tracklets_active = value

    @property
    def tracklets_finished(self):
        return self.state.tracklets_finished

    @tracklets_finished.setter
    def tracklets_finished(self, value):
        self.state.tracklets_finished = value

    @property
    def latest_detections(self):
        return self.state.latest_detections

    @latest_detections.setter
    def latest_detections(self, value):
        self.state.latest_detections = value

    @property
    def max_id(self):
        return self.state.max_id

    @max_id.setter
    def max_id(self, value):
        self.state.max_id = value

    @property
    def frame_num(self):
        return self.state.frame_num

    @frame_num.setter
    def frame_num(self, value):
        self.state.frame_num = value

    @property
    def frame(self):
        return self.state.frame

    @frame.setter
    def frame(self, value):
        self.state.frame = value

    @property
    def timestamp(self):
        return self.state.timestamp

    @timestamp.setter
    def timestamp(self, value):
        self.state.timestamp = value

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

        # Detection
        self.latest_detections = self.detect(img)

        # Encoding
        features = self.encode(self.latest_detections, img)

        # Data Association
        row_ind, col_ind = self.match(self.tracklets_active, features)

        # Tracklet Update
        self.update(row_ind, col_ind, self.latest_detections, features)

        # Log status
        self.log(frame_num=self.frame_num, dets=len(self.latest_detections), matches=len(row_ind),
                 targets=len(self.tracklets_active))

    @Timer.timer('det')
    def detect(self, img: np.ndarray) -> List[Detection]:
        detections = self.detector(img)
        if self.detection_filters is not None:
            for filter in self.detection_filters:
                detections = filter(detections)
        return detections

    @Timer.timer('assoc')
    def match(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        if len(detection_features) > 0:
            return self.matcher(tracklets, detection_features)
        else:
            return [], []

    @Timer.timer('enc')
    def encode(self, detections: List[Union[Detection, Prediction]], img: np.ndarray) -> List[Dict]:
        """
        Encode detections using all encoders.

        Args:
            detections: A list of Detection objects.
            img: The image ndarray.

        Returns:
            A list of dicts, with features generated by encoders for each detection.
        """
        features = [{'box': detections[i].box,
                     'score': detections[i].score,
                     'mask': detections[i].mask} for i in range(len(detections))]
        for encoder in self.encoders:
            _features = encoder(detections, img)
            for i in range(len(detections)):
                features[i][encoder.name] = _features[i]
        return features

    @Timer.timer('pred')
    def predict(self, img: np.ndarray) -> None:
        """
        Predict target positions in the incoming frame.

        Args:
            img: The image ndarray.
        """
        if self.predictor is not None:
            self.predictor(self.tracklets_active, img)

    @Timer.timer('upd')
    def update(self, row_ind: List[int], col_ind: List[int], detections: List[Detection],
               detection_features: List[Dict]) -> None:
        """
        Update the tracklets.
        *****************************************************
        Override this function for customized updating policy
        *****************************************************

        Args:
            row_ind: A list of integers. Indices of the matched tracklets.
            col_ind: A list of integers. Indices of the matched detections.
            detections: A list of Detection objects.
            detection_features: The features of the detections.
                By default it's a list of dictionaries, but it can be any form you want.
        """
        # Update tracked tracklets' features
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detections[col_ind[i]],
                                                     detection_features[col_ind[i]])

        # Deal with unmatched tracklets
        tracklets_to_kill = []
        unmatched_tracklets = []
        for i in range(len(self.tracklets_active)):
            if i not in row_ind:
                if self.tracklets_active[i].fade():
                    tracklets_to_kill.append(self.tracklets_active[i])
                else:
                    unmatched_tracklets.append(self.tracklets_active[i])

        # Kill tracklets that are unmatched for a while
        for tracklet in tracklets_to_kill:
            self.kill_tracklet(tracklet)

        # Create new tracklets with unmatched detections
        for i in range(len(detection_features)):
            new_tracklets = []
            if i not in col_ind:
                new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=self.max_ttl,
                                        max_feature_history=self.max_feature_history,
                                        max_detection_history=self.max_detection_history,
                                        min_time_lived=self.min_time_lived)
                new_tracklets.append(new_tracklet)
                self.add_tracklet(new_tracklet)
            if self.predictor is not None:
                self.predictor.initiate(new_tracklets)

    def terminate(self) -> None:
        """
        Terminate tracking and move all active tracklets to the finished ones.
        """
        for tracklet in self.tracklets_active:
            self.kill_tracklet(tracklet)

    def add_tracklet(self, tracklet: Tracklet) -> None:
        """
        Add a tracklet to the active tracklets after giving it a new ID.

        Args:
            tracklet: The tracklet to be added.
        """
        tracklet.id = self.max_id
        self.max_id += 1
        self.tracklets_active.append(tracklet)

    def kill_tracklet(self, tracklet: Tracklet) -> None:
        self.tracklets_active.remove(tracklet)
        tracklet.finished = True
        if self.keep_finished_tracks:
            if tracklet.time_lived >= self.min_time_lived:
                self.tracklets_finished.append(tracklet)
        else:
            del tracklet

    def log(self, frame_num: int, dets: int, matches: int, targets: int):
        logstr = 'Frame #{}: {} dets, {} matches, {} targets left. '.format(frame_num, dets, matches, targets)
        logstr += Timer.logstr()
        self.logger.info(logstr)


def build_tracker(cfg):
    return TRACKER_REGISTRY.get(cfg.type)(**(cfg.to_dict(ignore_keywords=False)))

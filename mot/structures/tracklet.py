import numpy as np
from typing import List, Dict, Tuple, Union

from mot.utils.hcc import hierarchical_cluster
from mot.structures import Detection, Prediction


class Tracklet:
    def __init__(self, id: int, frame_id: int, detection: Detection, feature: Dict, max_ttl: int = 30,
                 max_feature_history: int = 30, max_detection_history: int = 3000, min_time_lived: int = 1,
                 globalID: int = -1, cluster_frequency: int = 30, n_feature_samples: int = 8) -> None:
        # Tracklet ID number, usually starts from 1. Shouldn't be modified during online tracking.
        self.id: int = id
        # Tracklet global ID number in multi-camera tracking.
        self.globalID: int = globalID
        # Box coordinate of the last target position with (left, top, right, bottom).
        self.last_detection: Detection = detection
        # An array storing past features. Only keeping `max_history` frames.
        self.detection_history: List[Tuple[int, Detection]] = [(frame_id, detection)]
        # A feature dictionary.
        self.feature: Dict = feature
        # An array storing past features. Only keeping `max_history` frames.
        self.feature_history: List[Tuple[int, Dict]] = [(frame_id, feature)]
        # Max Time-to-Live. Tracklets will get killed if TTL times out.
        self.max_ttl: int = max_ttl
        # Parameter limiting the past history boxes to keep.
        self.max_detection_history: int = max_detection_history
        # Parameter limiting the past history features to keep.
        self.max_feature_history: int = max_feature_history
        # The actual Time-to-Live of a tracklet.
        self.ttl: int = max_ttl
        # The time lived (time matched with a measurement) of the tracklet.
        self.time_lived: int = 0
        # The motion prediction, if any.
        self.prediction: Union[None, Prediction] = None
        # Whether the target was just detected or not.
        self.detected: bool = True
        # The minimum time lived for a target to be confirmed.
        self.min_time_lived = min_time_lived
        # Whether the tracklet has been killed or not.
        self.finished = False
        # Starting frame num
        self.start_frame_num: int = frame_id
        # Latest active frame num
        self.last_active_frame_num: int = frame_id
        # Frequency for feature clustering
        self.cluster_frequency: int = cluster_frequency
        # Number of feature samples
        self.n_feature_samples: int = n_feature_samples
        # All sample features
        self.sample_features: Union[np.ndarray, None] = None

    def predict(self) -> np.ndarray:
        if self.prediction is not None:
            return self.prediction.box
        else:
            return self.last_detection.box

    def update(self, frame_id, detection, feature):
        self.detected = True
        self.last_detection = detection
        self.feature = feature
        self.last_active_frame_num = frame_id
        if len(self.feature_history) >= self.max_feature_history:
            self.feature_history.pop(0)
        if len(self.detection_history) >= self.max_detection_history:
            self.detection_history.pop(0)
        self.detection_history.append((frame_id, detection))
        self.feature_history.append((frame_id, feature))
        if frame_id % self.cluster_frequency == 0:
            self.update_sample_features()
        if self.ttl < self.max_ttl:
            self.ttl += 1
        self.time_lived += 1

    def update_sample_features(self):
        all_features = np.stack([feature['openreid'] for (i, feature) in self.feature_history], axis=0)
        all_features = np.vstack(
            (self.sample_features, all_features)) if self.sample_features is not None else all_features
        clusterIDs = hierarchical_cluster(all_features, t=self.n_feature_samples, linkage_method='average',
                                          criterion='maxclust')
        sample_features = []
        for clusterID in np.unique(clusterIDs):
            inds = np.where(clusterIDs == clusterID)[0]
            rand_ind = int(np.random.choice(inds))
            sample_features.append(all_features[rand_ind])

        self.sample_features = np.stack(sample_features)

    def fade(self):
        self.detected = False
        self.last_detection = Detection(self.predict(), 0)
        self.ttl -= 1
        return self.ttl <= 0

    def is_confirmed(self):
        return self.time_lived > self.min_time_lived

    def is_detected(self):
        return self.detected

    def is_finished(self):
        return self.finished

    def time_overlap(self, other):
        return max(0, min(self.last_active_frame_num, other.last_active_frame_num) - max(self.start_frame_num,
                                                                                         other.start_frame_num))

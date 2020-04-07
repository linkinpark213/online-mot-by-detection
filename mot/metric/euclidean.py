import logging
import numpy as np
from typing import Dict

from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, **kwargs):
        super(EuclideanMetric, self).__init__(**kwargs)

    def similarity(self, tracklet_feature: Dict, detection_feature: Dict):
        return 1 - self.distance(tracklet_feature, detection_feature)

    def distance(self, tracklet_feature, detection_feature):
        a = tracklet_feature[self.name]
        b = detection_feature[self.name]
        return np.square(np.linalg.norm(a - b))

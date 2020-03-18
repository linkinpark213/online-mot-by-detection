import logging
import numpy as np

from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, cfg):
        super(EuclideanMetric, self).__init__(cfg)

    def similarity(self, tracklet_feature, detection_feature):
        return 1 - self.distance(tracklet_feature, detection_feature)

    def distance(self, tracklet_feature, detection_feature):
        a = tracklet_feature[self.name]
        b = detection_feature[self.name]
        return np.square(np.linalg.norm(a - b))

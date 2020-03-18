import logging
import numpy as np

from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, cfg):
        super(CosineMetric, self).__init__(cfg)

    def similarity(self, tracklet_feature, detection_feature):
        a = tracklet_feature[self.name]
        b = detection_feature[self.name]
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-16)

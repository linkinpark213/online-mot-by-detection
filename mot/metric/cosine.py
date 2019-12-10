import logging
import numpy as np
import mot.utils.debug
from .metric import Metric


class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, encoding, history=1):
        super(CosineMetric, self).__init__(encoding, history)

    def distance(self, tracklet_feature, detection_feature):
        a = tracklet_feature[self.encoding]
        b = detection_feature[self.encoding]
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-16)

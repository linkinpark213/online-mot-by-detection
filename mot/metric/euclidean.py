import logging
import numpy as np
import mot.utils.debug
from .metric import Metric


class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, encoding, history=1):
        super(EuclideanMetric, self).__init__(encoding, history)

    def distance(self, tracklet_feature, detection_feature):
        a = tracklet_feature[self.encoding]
        b = detection_feature[self.encoding]
        return 1 - np.linalg.norm(a - b)

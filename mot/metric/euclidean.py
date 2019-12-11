import logging
import numpy as np
from .metric import Metric


class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, encoding, history=1):
        super(EuclideanMetric, self).__init__(encoding, history)

    def similarity(self, tracklet_feature, detection_feature):
        return 1 - self.distance(tracklet_feature, detection_feature)

    def distance(self, tracklet_feature, detection_feature):
        a = tracklet_feature[self.encoding]
        b = detection_feature[self.encoding]
        return np.square(np.linalg.norm(a - b))

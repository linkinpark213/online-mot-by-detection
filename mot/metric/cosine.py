import logging
import numpy as np
from typing import Dict

from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, **kwargs):
        super(CosineMetric, self).__init__(**kwargs)

    def similarity(self, tracklet_feature: Dict, detection_feature: Dict):
        a = tracklet_feature[self.name]
        b = detection_feature[self.name]
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-16)

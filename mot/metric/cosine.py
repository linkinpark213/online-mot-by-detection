import time
import logging
import numpy as np
from typing import Union

from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, **kwargs):
        super(CosineMetric, self).__init__(**kwargs)

    def similarity(self, tracklet_encoding: np.ndarray, detection_encoding: np.ndarray) -> Union[float, np.ndarray]:
        s = np.dot(tracklet_encoding, detection_encoding) / (
                (np.linalg.norm(tracklet_encoding) * np.linalg.norm(detection_encoding)) + 1e-16)
        return s

import logging
import numpy as np
from typing import Union

from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, **kwargs):
        super(EuclideanMetric, self).__init__(**kwargs)

    def similarity(self, tracklet_encoding: np.ndarray, detection_encoding: np.ndarray) -> Union[float, np.ndarray]:
        return 1 - np.square(np.linalg.norm(tracklet_encoding - detection_encoding))

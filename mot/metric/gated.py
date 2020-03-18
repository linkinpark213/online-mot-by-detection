import numpy as np
from typing import List, Dict, Union

from mot.structures import Tracklet
from .metric import Metric, METRIC_REGISTRY, build_metric


@METRIC_REGISTRY.register()
class GatedMetric(Metric):
    def __init__(self, cfg):
        self.original_metric = build_metric(cfg.metric)
        self.threshold = cfg.threshold
        self.encoding = self.original_metric.encoding + '_gated'
        super(GatedMetric, self).__init__(None)

    def affinity_matrix(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[
        np.ndarray, List[List[float]]]:
        matrix = self.original_metric(tracklets, detection_features)
        matrix[np.where(matrix < self.threshold)] = 0

        # For debugging
        self._log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def similarity(self, tracklet_feature: Dict, detection_feature: Dict) -> float:
        # To make this class not abstract
        return 0.0

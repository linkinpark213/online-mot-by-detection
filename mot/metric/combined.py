import logging
import numpy as np
from typing import List, Dict, Union

from mot.utils import Config
from mot.structures import Tracklet
from .metric import Metric, METRIC_REGISTRY, build_metric


@METRIC_REGISTRY.register()
class CombinedMetric(Metric):
    def __init__(self, metrics: List[Config], name: str = 'combined', **kwargs):
        self.metrics = [build_metric(metric_cfg) for metric_cfg in metrics]
        super(CombinedMetric, self).__init__(encoding='', name=name, **kwargs)

    def affinity_matrix(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[
        np.ndarray, List[List[float]]]:
        matrices = []

        for i in range(len(self.metrics)):
            matrix = self.metrics[i](tracklets, detection_features)
            matrices.append(matrix)

        matrices = np.array(matrices)
        matrix = np.zeros_like(matrices[0])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = self.combine(matrices[:, i, j])

        # For debugging
        self._log_affinity_matrix(matrix, tracklets, self.name)

        return matrix

    def similarity(self, tracklet_feature: Dict, detection_feature: Dict) -> float:
        return 0.0

    def combine(self, scores):
        raise NotImplementedError('Extend the CombinedMetric class to implement your own combination method.')


@METRIC_REGISTRY.register()
class ProductMetric(CombinedMetric):
    def combine(self, scores):
        return np.product(scores)


@METRIC_REGISTRY.register()
class SummationMetric(CombinedMetric):
    def combine(self, scores):
        return np.sum(scores)


@METRIC_REGISTRY.register()
class WeightedMetric(CombinedMetric):
    def __init__(self, metrics, weights):
        super().__init__(metrics)
        self.weights = weights

    def combine(self, scores):
        return np.sum([score * weight for (score, weight) in zip(scores, self.weights)])

import logging
import numpy as np
from .metric import Metric


class CombinedMetric(Metric):
    def __init__(self, metrics):
        self.metrics = metrics
        self.encoding = 'combined'
        super(CombinedMetric, self).__init__('combined', history=1)

    def __call__(self, tracklets, detection_features):
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
        self._log_affinity_matrix(matrix, tracklets, self.encoding)

        return matrix

    def combine(self, scores):
        raise NotImplementedError('Extend the CombinedMetric class to implement your own combination method.')


class ProductMetric(CombinedMetric):
    def combine(self, scores):
        return np.product(scores)


class SummationMetric(CombinedMetric):
    def combine(self, scores):
        return np.sum(scores)


class WeightedMetric(CombinedMetric):
    def __init__(self, metrics, weights):
        super().__init__(metrics)
        self.weights = weights

    def combine(self, scores):
        return np.sum([score * weight for (score, weight) in zip(scores, self.weights)])

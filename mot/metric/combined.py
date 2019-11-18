import logging
import numpy as np
import mot.utils.debug
from .metric import Metric


class CombinedMetric(Metric):
    def __init__(self, metrics):
        super(CombinedMetric).__init__()
        self.metrics = metrics
        self.encoding = 'combined'

    def __call__(self, tracklets, detection_features, img):
        matrices = []

        logger = logging.getLogger('MOT')

        for i in range(len(self.metrics)):
            matrix = self.metrics[i](tracklets, detection_features, img)

            # Log for debugging
            mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.metrics[i].encoding, logger)

            matrices.append(matrix)

        matrices = np.array(matrices)
        matrix = np.zeros_like(matrices[0])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = self.combine(matrices[:, i, j])

        # For debugging
        mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.encoding, logger)

        return matrix

    def combine(self, scores):
        raise NotImplementedError('Extend the CombinedMetric class to implement your own combination method.')


class ProductMetric(CombinedMetric):
    def combine(self, scores):
        return np.product(scores)


class SummationMetric(CombinedMetric):
    def combine(self, scores):
        return np.sum(scores)

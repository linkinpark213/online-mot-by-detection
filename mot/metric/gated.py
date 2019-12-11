import logging
import numpy as np
from .metric import Metric


class GatedMetric(Metric):
    def __init__(self, original_metric, gate_value):
        self.original_metric = original_metric
        self.gate_value = gate_value
        self.encoding = original_metric.encoding + '_gated'
        super(GatedMetric, self).__init__(self.encoding)

    def __call__(self, tracklets, detection_features):
        matrix = self.original_metric(tracklets, detection_features)
        matrix[np.where(matrix < self.gate_value)] = 0

        # For debugging
        self._log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

import logging
import numpy as np
import mot.utils.debug
from .metric import Metric


class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, encoding):
        super(CosineMetric).__init__()
        self.encoding = encoding

    def __call__(self, tracklets, detection_features):
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                matrix[i][j] = self.cos(tracklets[i].feature_history[-1][1][self.encoding],
                                        detection_features[j][self.encoding])
        mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def cos(self, a, b):
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-16)

import logging
import numpy as np
import mot.utils.debug
from .metric import Metric


class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, encoding, history=1):
        super(CosineMetric).__init__()
        self.encoding = encoding
        self.history = history

    def __call__(self, tracklets, detection_features):
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                affinities = []
                for k in range(min(self.history, len(tracklets[i].feature_history))):
                    affinities.append(self.cos(tracklets[i].feature_history[-k - 1][1][self.encoding],
                                               detection_features[j][self.encoding]))
                matrix[i][j] = sum(affinities) / len(affinities)
        mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def cos(self, a, b):
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-16)

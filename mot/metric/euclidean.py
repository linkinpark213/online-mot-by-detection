import logging
import numpy as np
import mot.utils.debug
from .metric import Metric


class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, encoding, history=1):
        super(EuclideanMetric).__init__()
        self.encoding = encoding
        assert history > 0, 'At least one step backward in history consideration'
        self.history = 1

    def __call__(self, tracklets, detection_features):
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                sum = 0
                if len(tracklets[i].feature_history) < self.history:
                    history = len(tracklets[i].feature_history)
                else:
                    history = self.history
                for k in range(history):
                    sum += self.euclidean(tracklets[i].feature_history[-k - 1][1][self.encoding][0],
                                          detection_features[j][self.encoding][0])
                matrix[i][j] = - sum / history
        mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.encoding, logging.getLogger('MOT'))
        return matrix

    def euclidean(self, a, b):
        return 1 - np.linalg.norm(a - b)

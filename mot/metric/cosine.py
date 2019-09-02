import numpy as np
from .metric import Metric


class CosineMetric(Metric):
    """
    An affinity metric that considers the cosine of angles between tracklets' features and detections' features.
    """

    def __init__(self, encoder):
        super(CosineMetric).__init__()
        self.encoder = encoder

    def __call__(self, tracklets, detected_boxes, img):
        matrix = np.zeros([len(tracklets), len(detected_boxes)])
        features = self.encoder(detected_boxes, img)
        for i in range(len(tracklets)):
            for j in range(len(detected_boxes)):
                matrix[i][j] = self.cos(tracklets[i].feature[0], features[j][0])
        return matrix, features

    def cos(self, a, b):
        return np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-16)

import numpy as np
from .metric import Metric


class EuclideanMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, encoder):
        super(EuclideanMetric).__init__()
        self.encoder = encoder
        self.name = encoder.name

    def __call__(self, tracklets, detected_boxes, img):
        matrix = np.zeros([len(tracklets), len(detected_boxes)])
        features = self.encoder(detected_boxes, img)
        for i in range(len(tracklets)):
            for j in range(len(detected_boxes)):
                matrix[i][j] = self.euclidean(tracklets[i].feature[self.encoder.name][0], features[j][0])


        for line in matrix:
            for num in line:
                print(num, end=' ')
            print('')


        return matrix, features

    def euclidean(self, a, b):
        return 1 - np.linalg.norm(a - b)

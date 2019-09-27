import numpy as np
from .metric import Metric
import torch


class MMMetric(Metric):
    """
    An affinity metric that considers the euclidean distances between tracklets' features and detections' features.
    """

    def __init__(self, encoder, history=1):
        super(MMMetric).__init__()
        self.encoder = encoder
        self.name = encoder.name
        assert history > 0, 'At least one step backward in history consideration'
        self.history = 1

    def __call__(self, tracklets, detected_boxes, img):
        matrix = np.zeros([len(tracklets), len(detected_boxes)])
        features = self.encoder(detected_boxes, img)
        for i in range(len(tracklets)):
            for j in range(len(detected_boxes)):
                sum = 0
                if len(tracklets[i].feature_history) < self.history:
                    history = len(tracklets[i].feature_history)
                else:
                    history = self.history
                for k in range(history):
                    sum += self.torch_mm(tracklets[i].feature_history[-k - 1][1][self.encoder.name].view(1, 1024),
                                         features[j].view(1024, 1))[0][0]
                matrix[i][j] = sum / history
        return matrix, features

    def torch_mm(self, a, b):
        return torch.mm(a, b)

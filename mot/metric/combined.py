import logging
import numpy as np
from .metric import Metric


class CombinedMetric(Metric):
    def __init__(self, metrics):
        super(CombinedMetric).__init__()
        self.metrics = metrics
        self.name = 'combined'

    def __call__(self, tracklets, detected_boxes, img):
        matrices = []
        all_features = []
        for i in range(len(self.metrics)):
            matrix, features = self.metrics[i](tracklets, detected_boxes, img)

            ################
            # For debugging
            ################
            print('Metric {}:'.format(self.metrics[i].name))
            for line in matrix:
                for i in line:
                    print('{}'.format(i), end=' ')
                print()
            ################

            matrices.append(matrix)
            all_features.append(features)

        matrices = np.array(matrices)
        matrix = np.zeros_like(matrices[0])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] = self.combine(matrices[:, i, j])

        feature_dict = {}
        for i in range(len(self.metrics)):
            if type(all_features[i]) is dict:
                for key in all_features[i].keys():
                    feature_dict[key] = all_features[i][key]
            else:
                feature_dict[self.metrics[i].name] = all_features[i]
        return matrix, feature_dict

    def combine(self, scores):
        raise NotImplementedError('Extend the CombinedMetric class to implement your own combination method.')


class ProductMetric(CombinedMetric):
    def combine(self, scores):
        return np.product(scores)


class SummationMetric(CombinedMetric):
    def combine(self, scores):
        return np.sum(scores)

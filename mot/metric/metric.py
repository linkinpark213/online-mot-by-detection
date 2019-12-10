import logging
import numpy as np
import mot.utils.debug


class Metric:
    def __init__(self, encoding, history=1):
        assert history > 0, 'At least one step backward in history consideration'
        self.encoding = encoding
        self.history = history

    def __call__(self, tracklets, detection_features):
        """
        Calculate the similarity matrix between tracklets and detections.
        :param tracklets: A list of active tracklets, assuming its size is (m).
        :param detection_features: A list of N feature dicts (encoded by detections).
        :return: A matrix of shape (m, n) and a list of detection features.
        """
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                affinities = []
                for k in range(min(self.history, len(tracklets[i].feature_history))):
                    affinities.append(self.distance(tracklets[i].feature_history[-k - 1][1], detection_features[j]))
                matrix[i][j] = sum(affinities) / len(affinities)
        mot.utils.debug.log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def distance(self, tracklet_feature, detection_feature):
        raise NotImplementedError('Extend the Metric class to implement your own distance metric.')

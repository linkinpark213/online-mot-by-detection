import numpy as np
from .matcher import Matcher


class GreedyMatcher(Matcher):
    """
    A greedy matching algorithm, re-implemented according to the IoU tracker paper.
    """

    def __init__(self, metric, sigma):
        super().__init__(metric)
        self.sigma = sigma

    def __call__(self, tracklets, detection_features):
        similarity_matrix = self.metric(tracklets, detection_features)
        row_ind = []
        col_ind = []
        for i in range(similarity_matrix.shape[0]):
            if len(similarity_matrix[i]) > 0:
                j = np.argmax(similarity_matrix[i])
                if similarity_matrix[i, j] > self.sigma:
                    if j not in col_ind:
                        row_ind.append(i)
                        col_ind.append(j)
        return row_ind, col_ind

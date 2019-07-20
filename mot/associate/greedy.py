import numpy as np
from .matcher import Matcher


class GreedyMatcher(Matcher):
    def __init__(self, sigma_iou):
        super(Matcher).__init__()
        self.sigma_iou = sigma_iou

    def __call__(self, similarity_matrix):
        row_ind = []
        col_ind = []
        for i in range(similarity_matrix.shape[0]):
            j = np.argmax(similarity_matrix[i])
            if similarity_matrix[i, j] > self.sigma_iou:
                if j not in col_ind:
                    row_ind.append(i)
                    col_ind.append(j)
        return row_ind, col_ind

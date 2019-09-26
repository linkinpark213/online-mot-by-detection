import numpy as np
import scipy.optimize
from .matcher import Matcher


class HungarianMatcher(Matcher):
    def __init__(self, metric, sigma):
        super().__init__(metric)
        self.sigma = sigma

    def __call__(self, tracklets, detection_boxes, img):
        similarity_matrix, features = self.metric(tracklets, detection_boxes, img)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(1 - similarity_matrix)
        rows_to_remove = []
        cols_to_remove = []
        for i in range(len(row_ind)):
            if similarity_matrix[row_ind[i], col_ind[i]] < self.sigma:
                rows_to_remove.append(row_ind[i])
                cols_to_remove.append(col_ind[i])
        row_ind = row_ind.tolist()
        col_ind = col_ind.tolist()
        for i in range(len(rows_to_remove)):
            row_ind.remove(rows_to_remove[i])
            col_ind.remove(cols_to_remove[i])
        return np.array(row_ind), np.array(col_ind), features

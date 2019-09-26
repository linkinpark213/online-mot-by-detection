import scipy.optimize
from .matcher import Matcher


class HungarianMatcher(Matcher):
    def __init__(self, metric, sigma):
        super().__init__(metric)
        self.sigma = sigma

    def __call__(self, tracklets, detection_boxes, img):
        similarity_matrix, features = self.metric(tracklets, detection_boxes, img)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(1 - similarity_matrix)
        indices_to_remove = []
        for i in range(len(row_ind)):
            if similarity_matrix[row_ind[i], col_ind[i]] < self.sigma:
                indices_to_remove.append(i)
        for index in indices_to_remove:
            row_ind.remove(row_ind[index])
            col_ind.remove(col_ind[index])
        return row_ind, col_ind, features

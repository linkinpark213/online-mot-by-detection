import scipy.optimize
from .matcher import Matcher


class HungarianMatcher(Matcher):
    def __init__(self, metric):
        super().__init__(metric)

    def __call__(self, tracklets, detection_boxes, img):
        similarity_matrix, features = self.metric(tracklets, detection_boxes, img)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(1 - similarity_matrix)
        return row_ind, col_ind, features

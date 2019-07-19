import scipy.optimize
from .matcher import Matcher


class HungarianMatcher(Matcher):
    def __init__(self):
        super(Matcher).__init__()

    def __call__(self, similarity_matrix):
        return scipy.optimize.linear_sum_assignment(1 - similarity_matrix)

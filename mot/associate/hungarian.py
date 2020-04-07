import numpy as np
import scipy.optimize
from typing import List, Dict, Tuple

from .matcher import Matcher, MATCHER_REGISTRY


@MATCHER_REGISTRY.register()
class HungarianMatcher(Matcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_association(self, tracklets: List, detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        similarity_matrix = self.metric(tracklets, detection_features)
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(1 - similarity_matrix)
        valid_inds = [similarity_matrix[row_ind[i], col_ind[i]] > self.sigma for i in range(len(row_ind))]
        row_ind = row_ind[valid_inds]
        col_ind = col_ind[valid_inds]
        return row_ind, col_ind

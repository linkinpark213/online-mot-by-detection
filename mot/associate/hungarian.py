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
        return row_ind, col_ind

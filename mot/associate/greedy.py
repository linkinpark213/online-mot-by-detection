import numpy as np
from typing import List, Dict, Tuple

from .matcher import Matcher, MATCHER_REGISTRY


@MATCHER_REGISTRY.register()
class GreedyMatcher(Matcher):
    """
    A greedy matching algorithm, re-implemented according to the IoU tracker paper.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_association(self, tracklets: List, detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        similarity_matrix = self.metric(tracklets, detection_features)
        row_ind = []
        col_ind = []
        for i in range(similarity_matrix.shape[0]):
            if len(similarity_matrix[i]) > 0:
                j = np.argmax(similarity_matrix[i])
                if similarity_matrix[i, j] > self.sigma and j not in col_ind:
                    row_ind.append(i)
                    col_ind.append(j)
        return row_ind, col_ind

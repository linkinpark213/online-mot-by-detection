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
        flattened = np.reshape(similarity_matrix, (-1))
        inds = list(reversed(np.argsort(flattened)))

        row_ind = []
        col_ind = []
        for ind in inds:
            if flattened[ind] < self.sigma:
                break
            row, col = ind // len(detection_features), ind % len(detection_features)
            if row not in row_ind and col not in col_ind:
                row_ind.append(row)
                col_ind.append(col)
        return row_ind, col_ind

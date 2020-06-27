import numpy as np
from typing import List, Dict, Tuple

from .matcher import Matcher, MATCHER_REGISTRY


@MATCHER_REGISTRY.register()
class GreedyMatcher(Matcher):
    """
    A greedy matching algorithm, considering the edges in the order of cost.
    """

    def __init__(self, confident: bool = True, **kwargs):
        # If not confident, do not make any match if any conflict exists.
        self.confident = confident
        super().__init__(**kwargs)

    def data_association(self, tracklets: List, detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        similarity_matrix = self.metric(tracklets, detection_features)
        flattened = np.reshape(similarity_matrix, (-1))
        inds = list(reversed(np.argsort(flattened)))

        row_ind = []
        col_ind = []
        for ind in inds:
            if flattened[ind] < self.threshold:
                break
            row, col = ind // len(detection_features), ind % len(detection_features)
            if not self.confident or (row not in row_ind and col not in col_ind):
                row_ind.append(row)
                col_ind.append(col)

        # Remove all matches with conflicts
        if not self.confident and (len(np.unique(row_ind)) != len(row_ind) or len(np.unique(col_ind)) != len(col_ind)):
            duplicates = self._find_duplicate_inds(row_ind) | self._find_duplicate_inds(col_ind)
            for ind in sorted(list(duplicates), reverse=True):
                row_ind.pop(ind)
                col_ind.pop(ind)

        return row_ind, col_ind

    def _find_duplicate_inds(self, arr):
        seen = {}
        duplicates = set()
        for i, num in enumerate(arr):
            if num in seen:
                duplicates.add(i)
                duplicates.add(seen[num])
            seen[num] = i
        return duplicates

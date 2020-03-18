from typing import List, Dict, Tuple

from mot.structures import Tracklet
from .matcher import Matcher, MATCHER_REGISTRY, build_matcher


@MATCHER_REGISTRY.register()
class CascadeMatcher(Matcher):
    def __init__(self, cfg):
        super().__init__(None)
        self.matchers = [build_matcher(matcher) for matcher in cfg.matchers]

    def data_association(self, tracklets: List, detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        all_row_ind = []
        all_col_ind = []
        for matcher in self.matchers:
            row_ind, col_ind = matcher(tracklets, detection_features)
            for i in range(len(row_ind)):
                row = row_ind[i]
                col = col_ind[i]
                if not row in all_row_ind and not col in all_col_ind:
                    all_row_ind.append(row)
                    all_col_ind.append(col)

        return all_row_ind, all_col_ind

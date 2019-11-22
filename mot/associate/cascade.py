from .matcher import Matcher


class CascadeMatcher(Matcher):
    def __init__(self, matchers):
        super().__init__(None)
        self.matchers = matchers

    def __call__(self, tracklets, detection_features):
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

from .matcher import Matcher


class CascadeMatcher(Matcher):
    def __init__(self, matchers):
        super().__init__(None)
        self.matchers = matchers

    def __call__(self, tracklets, detection_boxes, img):
        remaining_tracklets = tracklets.copy()
        all_features = []
        all_row_ind = []
        all_col_ind = []
        for matcher in self.matchers:
            row_ind, col_ind, features = matcher(remaining_tracklets, detection_boxes, img)
            all_features.append(features)
            tracklets_to_remove = []
            for i in range(len(row_ind)):
                row = row_ind[i]
                col = col_ind[i]
                if not col in all_col_ind:
                    tracklets_to_remove.append(remaining_tracklets[row])
                    all_row_ind.append(row)
                    all_col_ind.append(col)
            for tracklet in tracklets_to_remove:
                remaining_tracklets.remove(tracklet)

        # Generate feature dictionaries for the detections
        feature_dicts = []
        for i in range(len(detection_boxes)):
            feature_dict = {}
            for j in range(len(self.matchers)):
                feature_dict[self.matchers[j].metric.name] = all_features[j][i]
            feature_dicts.append(feature_dict)

        return all_row_ind, all_col_ind, feature_dicts

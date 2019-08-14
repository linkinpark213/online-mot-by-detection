import mot.associate
import mot.detect
import mot.metric
import mot.encode
import mot.predict
from mot.tracker import Tracker
from mot.tracklet import Tracklet
from utils.demo import run_demo


class SORTracker(Tracker):
    def __init__(self, detector, metric, matcher, predictor):
        super().__init__(detector, metric, matcher)
        self.predictor = predictor

    def update(self, row_ind, col_ind, detection_boxes, detection_features):
        unmatched_tracklets = []
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detection_boxes[i], detection_features[col_ind[i]])

        tracklets_to_kill = []
        for i in range(len(self.tracklets_active)):
            if i not in row_ind:
                if self.tracklets_active[i].fade():
                    tracklets_to_kill.append(self.tracklets_active[i])
                else:
                    unmatched_tracklets.append(self.tracklets_active[i])
        for tracklet in tracklets_to_kill:
            self.tracklets_active.remove(tracklet)
            self.tracklets_finished.append(tracklet)

        for i in range(len(detection_features)):
            if i not in col_ind:
                self.add_tracklet(Tracklet(0, detection_boxes[i], detection_features[i]))


if __name__ == '__main__':
    detector = mot.detect.YOLOv3Detector(conf_threshold=0.5)
    encoder = mot.encode.UVSRUVSEncoder()
    metric = mot.metric.IoUMetric()
    matcher = mot.associate.HungarianMatcher()
    # predictor = mot.predict.KalmanPredictor()
    predictor = None

    tracker = SORTracker(detector, metric, matcher, predictor)

    run_demo(tracker)

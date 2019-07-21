import mot.associate
import mot.detect
import mot.metric
import mot.encode
from mot.tracker import Tracker
from mot.tracklet import Tracklet
from utils.demo import run_demo


class NoPredictionTracklet(Tracklet):
    def __init__(self, id, feature, max_ttl=30):
        super().__init__(id, feature, max_ttl=max_ttl)


class IoUTracker(Tracker):
    def __init__(self, detector, encoder, metric, matcher, sigma_conf):
        super().__init__(detector, encoder, metric, matcher)
        self.sigma_conf = sigma_conf

    def update(self, row_ind, col_ind, detection_features):
        unmatched_tracklets = []
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detection_features[col_ind[i]])

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
                if detection_features[i][4] > self.sigma_conf:
                    self.add_tracklet(NoPredictionTracklet(0, detection_features[i]))


if __name__ == '__main__':
    detector = mot.detect.CenterNetDetector(conf_threshold=0.5)
    encoder = mot.encode.BoxCoordinateEncoder()
    metric = mot.metric.IoUMetric()
    matcher = mot.associate.GreedyMatcher(sigma=0.3)

    tracker = IoUTracker(detector, encoder, metric, matcher, sigma_conf=0.3)

    run_demo(tracker)

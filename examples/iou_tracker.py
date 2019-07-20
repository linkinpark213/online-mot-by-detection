import cv2
import mot.associate
import mot.detect
import mot.metric
import mot.encode
from mot.tracker import Tracker
from mot.tracklet import Tracklet
import utils.vis


class NoPredictionTracklet(Tracklet):
    def __init__(self, id, feature, max_ttl=30, sigma_h=0.6):
        super().__init__(id, feature, max_ttl)
        self.max_score = 0

    def predict(self):
        return self.feature


class IoUTracker(Tracker):
    def __init__(self, detector, encoder, metric, matcher, sigma_h):
        super().__init__(detector, encoder, metric, matcher)
        self.sigma_h = sigma_h

    def update(self, row_ind, col_ind, detection_features):
        unmatched_tracklets = []
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(detection_features[col_ind[i]])

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
                if detection_features[i][4] > self.sigma_h:
                    self.add_tracklet(NoPredictionTracklet(0, detection_features[i]))


if __name__ == '__main__':
    detector = mot.detect.CenterNetDetector(conf_threshold=0.3)
    encoder = mot.encode.BoxCoordinateEncoder()
    metric = mot.metric.IoUMetric()
    matcher = mot.associate.GreedyMatcher(sigma_iou=0.3)

    tracker = IoUTracker(detector, encoder, metric, matcher, sigma_h=0.3)

    capture = cv2.VideoCapture('/home/linkinpark213/Dataset/DukeMTMC/videos/camera8/00001.MTS')
    while True:
        ret, image = capture.read()
        if not ret:
            break
        boxes = tracker.tick(image)
        image = utils.vis.draw_tracklets(image, tracker.tracklets_active)
        cv2.imshow('Video', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

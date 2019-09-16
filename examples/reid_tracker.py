import mot.encode
import mot.metric
import mot.associate
from mot.tracker import Tracker
from mot.tracklet import Tracklet
from mot.utils.evaluate import evaluate_zhejiang


class ReIDTracker(Tracker):
    def __init__(self, detector, matcher, sigma_conf):
        super().__init__(detector, matcher)
        self.sigma_conf = sigma_conf

    def update(self, row_ind, col_ind, detection_boxes, detection_features):
        unmatched_tracklets = []
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detection_boxes[col_ind[i]],
                                                     detection_features[col_ind[i]])

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
                if detection_boxes[i][4] > self.sigma_conf:
                    self.add_tracklet(Tracklet(0, detection_boxes[i], detection_features[i]))


if __name__ == '__main__':
    # detector = mot.detect.HTCDetector(conf_threshold=0.5)
    detector = None
    encoder = mot.encode.PCBEncoder('mot/encode/PCB/model/')
    metric = mot.metric.EuclideanMetric(encoder)
    matcher = mot.associate.GreedyMatcher(metric, sigma=0.3)

    tracker = ReIDTracker(detector, matcher, sigma_conf=0.3)

    # run_demo(tracker)

    # evaluate_mot(tracker, '/mnt/nasbi/no-backups/datasets/object_tracking/MOT/MOT16/train')
    # evaluate_mot(tracker, '/mnt/nasbi/no-backups/datasets/object_tracking/MOT/MOT16/test')

    evaluate_zhejiang(tracker, '/home/linkinpark213/Dataset/Zhejiang',
                             'data/det/HTC', show_result=True)

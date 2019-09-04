import mot.associate
import mot.detect
import mot.metric
import mot.encode
import mot.predict
from mot.tracker import Tracker
from mot.tracklet import Tracklet
from utils.demo import run_demo
from utils.evaluate import evaluate_mot_online, evaluate_zhejiang_online


class DeepSORTTracker(Tracker):
    def __init__(self, detector, matcher, predictor):
        super().__init__(detector, matcher)
        self.predictor = predictor

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
                self.add_tracklet(Tracklet(0, detection_boxes[i], detection_features[i], self.predictor))


if __name__ == '__main__':
    # detector = mot.detect.HTCDetector(conf_threshold=0.5)
    detector = None
    iou_metric = mot.metric.IoUMetric()
    iou_matcher = mot.associate.GreedyMatcher(iou_metric, sigma=0.3)
    reid_encoder = mot.encode.PCBEncoder('mot/encode/PCB/model/')
    reid_metric = mot.metric.EuclideanMetric(reid_encoder)
    reid_metric = mot.metric.ProductMetric((reid_metric, iou_metric))
    reid_matcher = mot.associate.GreedyMatcher(reid_metric, sigma=0.3)
    matcher = mot.associate.CascadeMatcher((reid_matcher, iou_matcher))
    predictor = mot.predict.KalmanPredictor()

    tracker = DeepSORTTracker(detector, matcher, predictor)

    # run_demo(tracker)

    # evaluate_mot(tracker, '/mnt/nasbi/no-backups/datasets/object_tracking/MOT/MOT16/train')
    # evaluate_mot(tracker, '/mnt/nasbi/no-backups/datasets/object_tracking/MOT/MOT16/test')

    evaluate_zhejiang_online(tracker, '/home/linkinpark213/Dataset/Zhejiang',
                             'data/det/HTC', show_result=True)
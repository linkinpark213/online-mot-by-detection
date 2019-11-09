import mot.detect
import mot.metric
import mot.associate
from mot.tracker import Tracker
from mot.tracklet import Tracklet


class CustomTracker(Tracker):
    def __init__(self, sigma_conf=0.3):
        detector = mot.detect.Detectron(
            '/home/linkinpark213/Source/detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',
            'detectron2://COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl')
        metric = mot.metric.IoUMetric()
        matcher = mot.associate.GreedyMatcher(metric, sigma=0.3)
        super().__init__(detector, matcher)
        self.sigma_conf = sigma_conf

    def update(self, row_ind, col_ind, detections, detection_features):
        unmatched_tracklets = []
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detections[col_ind[i]],
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
                if detection_features[i][4] > self.sigma_conf:
                    self.add_tracklet(Tracklet(0, self.frame_num, detections[i], detection_features[i]))

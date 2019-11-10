import mot.associate
import mot.encode
import mot.metric
import mot.detect
import mot.predict
from mot.tracker import Tracker
from mot.tracklet import Tracklet


class CustomTracker(Tracker):
    def __init__(self):
        detector = mot.detect.Detectron(
            '/home/linkinpark213/Source/detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',
            'detectron2://COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl')
        iou_metric = mot.metric.IoUMetric()
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.3)

        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet/outputs/checkpoints/')

        reid_metric = mot.metric.MMMetric(reid_encoder, history=10)
        combined_metric = mot.metric.ProductMetric((reid_metric, iou_metric))
        combined_matcher = mot.associate.HungarianMatcher(combined_metric, sigma=0.5)

        matcher = mot.associate.CascadeMatcher((combined_matcher, iou_matcher))
        # self.predictor = mot.predict.KalmanPredictor(box_type='xyxy', predict_type='xywh')
        super().__init__(detector, matcher)

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
                self.add_tracklet(
                    Tracklet(0, self.frame_num, detections[i], detection_features[i], self.predictor))

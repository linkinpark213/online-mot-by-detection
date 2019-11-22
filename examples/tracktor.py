import mot.encode
import mot.metric
import mot.detect
import mot.predict
import numpy as np
import mot.associate
import mot.utils.box
from mot.tracker import Tracker, Tracklet


class CustomTracker(Tracker):
    def __init__(self, sigma_active=0.5, lambda_active=0.6, lambda_new=0.3):
        detector = mot.detect.Detectron(
            'https://raw.githubusercontent.com/facebookresearch/detectron2/22e04d1432363be727797a081e3e9d48981f5189/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml',
            'detectron2://Misc/cascade_mask_rcnn_R_50_FPN_3x/144998488/model_final_480dd8.pkl')
        iou_metric = mot.metric.IoUMetric(use_prediction=True)
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.5)

        matcher = iou_matcher
        predictor = mot.predict.DetectronRCNNPredictor(
            'https://raw.githubusercontent.com/facebookresearch/detectron2/22e04d1432363be727797a081e3e9d48981f5189/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl'
        )
        # self.predictor = None
        self.sigma_active = sigma_active
        self.lambda_active = lambda_active
        self.lambda_new = lambda_new
        super().__init__(detector, [], matcher, predictor)

    def update(self, row_ind, col_ind, detections, detection_features):
        """
        Update the tracklets.
        :param row_ind: A list of integers. Indices of the matched tracklets.
        :param col_ind: A list of integers. Indices of the matched detections.
        :param detection_boxes: A list of Detection objects.
        :param detection_features: The features of the detections. It can be any form you want.
        """
        # Update tracked tracklets' features
        for i in range(len(row_ind)):
            tracklet = self.tracklets_active[row_ind[i]]
            tracklet.update(self.frame_num, tracklet.prediction, {'box': tracklet.prediction.box})

        # Deal with unmatched tracklets
        for i, tracklet in enumerate(self.tracklets_active):
            if tracklet.prediction.score < self.sigma_active:
                if tracklet.fade():
                    self.kill_tracklet(tracklet)

        # Kill tracklets with lower scores using NMS
        for i, tracklet in enumerate(self.tracklets_active):
            ious = mot.utils.box.iou(tracklet.prediction.box, [t.prediction.box for t in self.tracklets_active])
            overlapping_boxes = np.argwhere(ious > self.lambda_active)[0]
            for j in overlapping_boxes:
                if i == j:
                    continue
                else:
                    if tracklet.prediction.score >= self.tracklets_active[j].prediction.score:
                        self.kill_tracklet((self.tracklets_active[j]))
                    else:
                        self.kill_tracklet(tracklet)
                        break

        # Update tracklets
        for tracklet in self.tracklets_active:
            tracklet.last_detection.box = tracklet.prediction.box

        # Remove matched detections
        detections_to_remove = []
        for i, detection in enumerate(detections):
            if i not in col_ind:
                for tracklet in self.tracklets_active:
                    if mot.utils.box.iou(detection.box, tracklet.last_detection.box) > self.lambda_new:
                        detections_to_remove.append(detection)
                        break
            else:
                detections_to_remove.append(detection)
        for detection in detections_to_remove:
            detections.remove(detection)

        # Initiate new tracklets
        for i, detection in enumerate(detections):
            new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=1)
            self.add_tracklet(new_tracklet)
            self.predictor.initiate([new_tracklet])

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
        # detector = mot.detect.Detectron(
        #     'https://raw.githubusercontent.com/facebookresearch/detectron2/22e04d1432363be727797a081e3e9d48981f5189/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        #     'detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl')
        detector = mot.detect.MMDetector(
            '/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
            'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth'
        )
        iou_metric = mot.metric.IoUMetric(use_prediction=True)
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.5)

        matcher = iou_matcher
        # predictor = mot.predict.DetectronRCNNPredictor(
        #     'https://raw.githubusercontent.com/facebookresearch/detectron2/22e04d1432363be727797a081e3e9d48981f5189/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        #     'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl'
        # )
        # predictor = mot.predict.DetectronRCNNPredictor(detector)
        predictor = mot.predict.MMTwoStagePredictor(detector)
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
            tracklet.update(self.frame_num, tracklet.prediction,
                            {'box': tracklet.prediction.box, **detection_features[col_ind[i]]})

        # Deal with unmatched tracklets
        for i, tracklet in enumerate(self.tracklets_active):
            if tracklet.prediction.score < self.sigma_active:
                if tracklet.fade():
                    self.kill_tracklet(tracklet)

        # Kill tracklets with lower scores using NMS
        tracklets_to_kill = []
        for i, tracklet in enumerate(self.tracklets_active):
            ious = mot.utils.box.iou(tracklet.prediction.box, [t.prediction.box for t in self.tracklets_active])
            overlapping_boxes = np.argwhere(ious > self.lambda_active)
            for j in overlapping_boxes:
                if i == j[0]:
                    continue
                else:
                    if tracklet.prediction.score >= self.tracklets_active[j[0]].prediction.score:
                        tracklets_to_kill.append(self.tracklets_active[j[0]])
                    else:
                        tracklets_to_kill.append(tracklet)
                        break
        for tracklet in tracklets_to_kill:
            self.kill_tracklet(tracklet)

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

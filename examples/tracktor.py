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
        detector = mot.detect.MMDetector(
            '/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
            'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth'
        )
        iou_metric = mot.metric.IoUMetric(use_prediction=True)
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.5)

        matcher = iou_matcher
        predictor = mot.predict.MMTwoStagePredictor(detector)
        self.sigma_active = sigma_active
        self.lambda_active = lambda_active
        self.lambda_new = lambda_new
        self.tracklets_inactive = []
        super().__init__(detector, [], matcher, predictor)

    def tick(self, img):
        """
        Tracktor++ has to encode predictions besides the matched detections.
        :param img: A 3D numpy array with shape (H, W, 3). The new frame in the sequence.
        """
        self.frame_num += 1

        # Prediction
        self.predict(img)

        # Detection
        detections = self.detector(img)

        # Encoding
        features = [{'box': detections[i].box} for i in range(len(detections))]

        # Data Association
        row_ind, col_ind = self.matcher(self.tracklets_active, features)

        # Primary matching: Active tracklets and detections
        self.update_step_1(row_ind, col_ind, detections, features)

        # After the primary matching, update tracklets' features
        features = self.encode([tracklet.prediction for tracklet in self.tracklets_active], img)
        for i, tracklet in enumerate(self.tracklets_active):
            tracklet.update(self.frame_num, tracklet.last_detection,
                            {'box': tracklet.last_detection.box, **features[i]})

        # Remove matched detections and proceed to the secondary matching
        detections_to_remove = []
        for col in col_ind:
            detections_to_remove.append(detections[col])
        for detection in detections_to_remove:
            detections.remove(detection)

        # Secondary matching: Inactive tracklets and remaining detections
        if hasattr(self, 'secondary_matcher') and self.secondary_matcher is not None:
            features = self.encode(detections, img)
            new_row_ind, new_col_ind = self.secondary_matcher(self.tracklets_inactive, features)
            self.update_step_2(new_row_ind, new_col_ind, detections, features)
        else:
            self.initiate_new_tracklets(detections, features)

        self.logger.info(
            'Frame #{}: {} target(s) active, {} new detections'.format(self.frame_num, len(self.tracklets_active),
                                                                       len(detections)))

    def down_tracklet(self, tracklet):
        tracklet.ttl = self.max_ttl
        self.tracklets_active.remove(tracklet)
        self.tracklets_inactive.append(tracklet)

    def revive_tracklet(self, tracklet):
        self.tracklets_inactive.remove(tracklet)
        self.tracklets_active.append(tracklet)

    def terminate_tracklet(self, tracklet):
        self.tracklets_inactive.remove(tracklet)
        self.tracklets_finished.append(tracklet)

    def update_step_1(self, row_ind, col_ind, detections, detection_features):
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
            tracklet.last_detection.box = tracklet.prediction.box

        # Deal with unmatched tracklets
        for i, tracklet in enumerate(self.tracklets_active):
            if tracklet.prediction.score < self.sigma_active:
                self.down_tracklet(tracklet)

        # Kill tracklets with lower scores using NMS
        tracklets_to_dismiss = []
        for i, tracklet in enumerate(self.tracklets_active):
            ious = mot.utils.box.iou(tracklet.prediction.box, [t.prediction.box for t in self.tracklets_active])
            overlapping_boxes = np.argwhere(ious > self.lambda_active)
            for j in overlapping_boxes:
                if i == j[0]:
                    continue
                else:
                    if tracklet.prediction.score >= self.tracklets_active[j[0]].prediction.score:
                        if self.tracklets_active[j[0]] not in tracklets_to_dismiss:
                            tracklets_to_dismiss.append(self.tracklets_active[j[0]])
                    else:
                        if tracklet not in tracklets_to_dismiss:
                            tracklets_to_dismiss.append(tracklet)
                            break
        for tracklet in tracklets_to_dismiss:
            self.down_tracklet(tracklet)

        # Update tracklets
        for tracklet in self.tracklets_active:
            tracklet.last_detection.box = tracklet.prediction.box

    def update_step_2(self, row_ind, col_ind, detections, detection_features):
        # Revive matched inactive tracklets
        tracklets_to_revive = []
        for i in range(len(row_ind)):
            tracklet = self.tracklets_inactive[row_ind[i]]
            tracklet.update(self.frame_num, detections[col_ind[i]], detection_features[col_ind[i]])
            tracklets_to_revive.append(tracklet)
        for tracklet in tracklets_to_revive:
            self.revive_tracklet(tracklet)

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

        # Kill inactive tracklets that expires TTL
        for i, tracklet in enumerate(self.tracklets_inactive):
            if tracklet.fade():
                self.terminate_tracklet(tracklet)

        # Initiate new tracklets
        self.initiate_new_tracklets(detections, detection_features)

    def initiate_new_tracklets(self, detections, detection_features):
        for i, detection in enumerate(detections):
            new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=1)
            self.add_tracklet(new_tracklet)
            self.predictor.initiate([new_tracklet])

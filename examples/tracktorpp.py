import mot.encode
import mot.metric
import numpy as np
import mot.associate
import mot.utils.box
from mot.tracklet import Tracklet
from examples.tracktor import CustomTracker as Tracktor


class CustomTracker(Tracktor):
    def __init__(self, sigma_active=0.5, lambda_active=0.6, lambda_new=0.3, max_ttl=30):
        super().__init__(sigma_active=sigma_active, lambda_active=lambda_active, lambda_new=lambda_new)
        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet')
        reid_metric = mot.metric.CosineMetric(reid_encoder.name, history=30)
        reid_metric = mot.metric.GatedMetric(reid_metric, 0.7)
        self.secondary_matcher = mot.associate.HungarianMatcher(reid_metric, 0.9)
        self.encoders.append(reid_encoder)
        self.max_ttl = max_ttl

    # def update(self, row_ind, col_ind, detections, detection_features):
    #     """
    #     Update the tracklets. Tracktor introduces a new MOT paradigm and involves additional matching steps.
    #     :param row_ind: A list of integers. Indices of the matched tracklets.
    #     :param col_ind: A list of integers. Indices of the matched detections.
    #     :param detection_boxes: A list of Detection objects.
    #     :param detection_features: The features of the detections. It can be any form you want.
    #     """
    #     # Update tracked tracklets' features
    #     for i in range(len(row_ind)):
    #         tracklet = self.tracklets_active[row_ind[i]]
    #         tracklet.update(self.frame_num, tracklet.prediction, {'box': tracklet.prediction.box})
    #
    #     # Deal with unmatched tracklets
    #     for i, tracklet in enumerate(self.tracklets_active):
    #         if tracklet.prediction.score < self.sigma_active:
    #             self.kill_tracklet(tracklet)
    #
    #     # Kill tracklets with lower scores using NMS
    #     tracklets_to_kill = []
    #     for i, tracklet in enumerate(self.tracklets_active):
    #         ious = mot.utils.box.iou(tracklet.prediction.box, [t.prediction.box for t in self.tracklets_active])
    #         overlapping_boxes = np.argwhere(ious > self.lambda_active)
    #         for j in overlapping_boxes:
    #             if i == j[0]:
    #                 continue
    #             else:
    #                 if tracklet.prediction.score >= self.tracklets_active[j[0]].prediction.score:
    #                     if self.tracklets_active[j[0]] not in tracklets_to_kill:
    #                         tracklets_to_kill.append(self.tracklets_active[j[0]])
    #                 else:
    #                     if tracklet not in tracklets_to_kill:
    #                         tracklets_to_kill.append(tracklet)
    #                         break
    #     for tracklet in tracklets_to_kill:
    #         self.kill_tracklet(tracklet)
    #
    #     # Update tracklets
    #     for tracklet in self.tracklets_active:
    #         tracklet.last_detection.box = tracklet.prediction.box
    #
    #     # Match detections with inactive tracklets
    #     if hasattr(self, 'secondary_matcher'):
    #         new_row_ind, new_col_ind = self.secondary_matcher(self.tracklets_inactive, detection_features)
    #         tracklets_to_revive = []
    #         for i in range(len(new_row_ind)):
    #             if new_col_ind[i] not in col_ind:
    #                 tracklet = self.tracklets_inactive[new_row_ind[i]]
    #                 tracklet.update(self.frame_num, detections[new_col_ind[i]], detection_features[new_col_ind[i]])
    #                 tracklets_to_revive.append(tracklet)
    #         for tracklet in tracklets_to_revive:
    #             self.kill_tracklet(tracklet)
    #         col_ind = col_ind + new_col_ind
    #
    #     # Remove matched detections
    #     detections_to_remove = []
    #     for i, detection in enumerate(detections):
    #         if i not in col_ind:
    #             for tracklet in self.tracklets_active:
    #                 if mot.utils.box.iou(detection.box, tracklet.last_detection.box) > self.lambda_new:
    #                     detections_to_remove.append(detection)
    #                     break
    #         else:
    #             detections_to_remove.append(detection)
    #     for detection in detections_to_remove:
    #         detections.remove(detection)
    #
    #     # Kill inactive tracklets that expires TTL
    #     for i, tracklet in enumerate(self.tracklets_inactive):
    #         if tracklet.fade():
    #             self.tracklets_inactive.remove(tracklet)
    #             self.tracklets_finished.append(tracklet)
    #
    #     # Initiate new tracklets
    #     for i, detection in enumerate(detections):
    #         new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=self.max_ttl)
    #         self.add_tracklet(new_tracklet)
    #         self.predictor.initiate([new_tracklet])

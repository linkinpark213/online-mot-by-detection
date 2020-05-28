import numpy as np
from typing import List

import mot.encode
import mot.metric
import mot.detect
import mot.predict
import mot.associate
import mot.utils.box
from mot.utils import Config
from mot.structures import Tracklet
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.associate import build_matcher
from mot.predict import build_predictor
from .tracker import Tracker, TRACKER_REGISTRY


@TRACKER_REGISTRY.register()
class Tracktor(Tracker):
    def __init__(self, detector: Config, matcher: Config, encoders: List[Config], predictor: Config,
                 sigma_active: float = 0.5, lambda_active: float = 0.6,
                 lambda_new: float = 0.3, secondary_matcher: Config = None, **kwargs):
        detector = build_detector(detector)
        matcher = build_matcher(matcher)
        encoders = [build_encoder(encoder_cfg) for encoder_cfg in encoders]
        predictor = build_predictor(predictor)
        self.sigma_active = sigma_active
        self.lambda_active = lambda_active
        self.lambda_new = lambda_new
        self.tracklets_inactive = []
        if secondary_matcher is not None:
            self.secondary_matcher = build_matcher(secondary_matcher)
        super().__init__(detector, encoders, matcher, predictor, **kwargs)

    def tick(self, img):
        """
        Tracktor++ has to encode predictions besides the matched detections.
        :param img: A 3D numpy array with shape (H, W, 3). The new frame in the sequence.
        """
        self.frame_num += 1

        # Prediction
        self.predict(img)

        # Detection
        detections = self.detect(img)

        # Encoding
        features = self.encode(detections, img)

        # Data Association
        row_ind, col_ind = self.match(self.tracklets_active, features)

        self.log_step_1(self.tracklets_active, detections, row_ind, col_ind)

        # Primary matching: Active tracklets and detections (and detections are less relied on)
        self.update_step_1(row_ind, col_ind, detections, features)

        # After the primary matching, update tracklets' features
        tracklet_features = self.encode([tracklet.prediction for tracklet in self.tracklets_active], img)
        for i, tracklet in enumerate(self.tracklets_active):
            tracklet.update(self.frame_num, tracklet.last_detection,
                            {'box': tracklet.last_detection.box, **tracklet_features[i]})

        # Remove matched detections and proceed to the secondary matching
        detections_to_remove = []
        for col in col_ind:
            detections_to_remove.append(detections[col])
        for detection in detections_to_remove:
            detections.remove(detection)

        # Secondary matching: Inactive tracklets and remaining detections
        features = self.encode(detections, img)
        if hasattr(self, 'secondary_matcher') and self.secondary_matcher is not None:
            new_row_ind, new_col_ind = self.secondary_matcher(self.tracklets_inactive, features)
            self.log_step_2(self.tracklets_inactive, detections, new_row_ind, new_col_ind)
            self.update_step_2(new_row_ind, new_col_ind, detections, features)
        else:
            self.initiate_new_tracklets(detections, features, [])

        self.logger.info(
            'Frame #{}: {} target(s) active, {} new detections'.format(self.frame_num, len(self.tracklets_active),
                                                                       len(detections)))

    def down_tracklet(self, tracklet):
        """
        Set a tracklet to temporarily inactive.
        Args:
            tracklet: A Tracklet object. The tracklet to set temporarily inactive.
        """
        tracklet.ttl = self.max_ttl
        self.tracklets_active.remove(tracklet)
        self.tracklets_inactive.append(tracklet)

    def revive_tracklet(self, tracklet):
        """
        Bring an inactive tracklet back to active tracklet list.
        Args:
            tracklet: A Tracklet object. The tracklet to bring back.
        """
        self.tracklets_inactive.remove(tracklet)
        self.tracklets_active.append(tracklet)

    def terminate_tracklet(self, tracklet):
        """
        Permanently kill a tracklet. This tracklet should be in the inactive list.
        Args:
            tracklet: A Tracklet object. The tracklet to kill.
        """
        self.tracklets_inactive.remove(tracklet)
        self.tracklets_finished.append(tracklet)

    def log_step_1(self, tracklets, detections, row_ind, col_ind):
        # Start with current situation
        self.logger.info(
            'Frame #{}: {} target(s) active, {} object(s) detected'.format(self.frame_num, len(self.tracklets_active),
                                                                           len(detections)))

        # And strong tracklets
        if len(row_ind) > 0:
            self.logger.debug('Tracklets that successfully predicted new boxes:')
            for i, row in enumerate(row_ind):
                box = tracklets[i].prediction.box
                self.logger.debug(
                    '\tTracklet #{:d}: l = {:.2f}, \tt = {:.2f}, \tr = {:.2f}, \tb = {:.2f}'.format(tracklets[i].id,
                                                                                                    box[0], box[1],
                                                                                                    box[2], box[3]))

    def update_step_1(self, row_ind, col_ind, detections, detection_features):
        """
        Update the tracklets that successfully predicted their new position.

        Args:
            row_ind: A list of integers. Indices of the matched tracklets.
            col_ind: A list of integers. Indices of the matched detections.
            detections: A list of Detection objects.
            detection_features: The features of the detections. It can be any form you want.
        """
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

    def log_step_2(self, tracklets, detections, row_ind, col_ind):
        # And detections
        if len(detections) > 0:
            self.logger.debug('Remaining detections:')
            for i, detection in enumerate(detections):
                box = detection.box
                self.logger.debug(
                    '\t#{:d}: l = {:.2f}, \tt = {:.2f}, \tr = {:.2f}, \tb = {:.2f}'.format(i, box[0], box[1], box[2],
                                                                                           box[3]))

        # And matches
        if len(row_ind) > 0:
            self.logger.debug('Matches:')
            for i, row in enumerate(row_ind):
                self.logger.debug('\tTracklet #{:d} -- Detection #{:d}'.format(tracklets[row].id, col_ind[i]))

    def log_initiate(self, detections, col_ind):
        if len(col_ind > 0):
            self.logger.debug('New tracklets:')
            count = 0
            for i, col in enumerate(detections):
                if i not in col_ind:
                    count += 1
                    self.logger.debug(
                        '\tTracklet #{:d}, from Detection #{:d}'.format(self.max_id + count + 1, col_ind[i]))

    def update_step_2(self, row_ind, col_ind, detections, detection_features):
        """
        Update inactive tracklets according to their matches with remaining detections.

        Args:
            row_ind: A list of integers. Indices of the matched tracklets.
            col_ind: A list of integers. Indices of the matched detections.
            detections: A list of Detection objects.
            detection_features: The features of the detections. It can be any form you want.
        """
        # Revive matched inactive tracklets
        tracklets_to_revive = []
        for i in range(len(row_ind)):
            tracklet = self.tracklets_inactive[row_ind[i]]
            tracklet.update(self.frame_num, detections[col_ind[i]], detection_features[col_ind[i]])
            tracklets_to_revive.append(tracklet)
        for tracklet in tracklets_to_revive:
            self.revive_tracklet(tracklet)

        # Remove matched detections
        for i, detection in enumerate(detections):
            if i not in col_ind:
                for tracklet in self.tracklets_active:
                    if mot.utils.box.iou(detection.box, tracklet.last_detection.box) > self.lambda_new:
                        col_ind.append(i)
                        break

        # Kill inactive tracklets that expires TTL
        for i, tracklet in enumerate(self.tracklets_inactive):
            if tracklet.fade():
                self.terminate_tracklet(tracklet)

        self.initiate_new_tracklets(detections, detection_features, col_ind)

    def initiate_new_tracklets(self, detections, detection_features, matched_col_ind):
        # Initiate new tracklets
        if len(detections) - len(matched_col_ind) > 0:
            self.logger.debug('New tracklets:')
        for i, detection in enumerate(detections):
            new_tracklets = []
            count = 0
            if i not in matched_col_ind:
                self.logger.debug(
                    '\tTracklet #{:d}, from Detection #{:d}'.format(self.max_id + count, i))
                count += 1
                new_tracklet = Tracklet(0, self.frame_num, detections[i], detection_features[i], max_ttl=1)
                self.add_tracklet(new_tracklet)
            if self.predictor is not None:
                self.predictor.initiate(new_tracklets)

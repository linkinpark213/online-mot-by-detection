import cv2
import logging
import numpy as np
from .tracklet import Tracklet


class Tracker:
    def __init__(self, detector, encoders, matcher):
        self.detector = detector
        self.encoders = encoders
        self.matcher = matcher
        self.max_id = 0
        self.tracklets_active = []
        self.tracklets_finished = []
        self.frame_num = 0

    def clear(self):
        self.max_id = 0
        self.tracklets_active = []
        self.tracklets_finished = []
        self.frame_num = 0

    def tick(self, img):
        """
        The tracker works online. For each new frame, the tracker ticks once.
        :param img: A 3D numpy array with shape (H, W, 3). The new frame in the sequence.
        """
        self.frame_num += 1

        # Detection
        detections = self.detector(img)

        # Encoding
        features = self.encode(detections, img)

        # Data Association
        row_ind, col_ind = self.matcher(self.tracklets_active, features, img)

        # Tracklet Update
        self.update(row_ind, col_ind, detections, features)

        logging.info(
            'Frame #{}: {} target(s) active, {} object(s) detected'.format(self.frame_num, len(self.tracklets_active),
                                                                           len(detections)))

    def encode(self, detections, img):
        """
        Encode detections using all encoders.
        :param detections: A list of Detection objects.
        :param img: The image ndarray.
        :return: A list of dicts, with features generated by encoders for each detection.
        """
        features = [{'box': detections[i].box} for i in range(len(detections))]
        for encoder in self.encoders:
            _features = encoder(detections, img)
            for i in range(len(detections)):
                features[i][encoder.name] = _features[i]
        return features

    def update(self, row_ind, col_ind, detections, detection_features):
        """
        Update the tracklets.
        *****************************************************
        Override this function for customized updating policy
        *****************************************************
        :param row_ind: A list of integers. Indices of the matched tracklets.
        :param col_ind: A list of integers. Indices of the matched detections.
        :param detection_boxes: A list of Detection objects.
        :param detection_features: The features of the detections. It can be any form you want.
        """
        # Update tracked tracklets' features
        for i in range(len(row_ind)):
            self.tracklets_active[row_ind[i]].update(self.frame_num, detections[col_ind[i]],
                                                     detection_features[col_ind[i]])

        # Deal with unmatched tracklets
        tracklets_to_kill = []
        unmatched_tracklets = []
        for i in range(len(self.tracklets_active)):
            if i not in row_ind:
                if self.tracklets_active[i].fade():
                    tracklets_to_kill.append(self.tracklets_active[i])
                else:
                    unmatched_tracklets.append(self.tracklets_active[i])

        # Kill tracklets that are unmatched for a while
        for tracklet in tracklets_to_kill:
            self.tracklets_active.remove(tracklet)
            self.tracklets_finished.append(tracklet)

        # Create new tracklets with unmatched detections
        for i in range(len(detection_features)):
            if i not in col_ind:
                self.add_tracklet(
                    Tracklet(0, self.frame_num, detections[i], detection_features[i], None))

    def assignment_matrix(self, similarity_matrix):
        """
        Calculate assignment matrix using the matching algorithm. Only for debugging.
        :param similarity_matrix: A 2D numpy array. The similarity matrix.
        :return: A 2D numpy array with the same shape as the similarity matrix. The assignment matrix.
        """
        row_ind, col_ind = self.matcher(similarity_matrix)
        assignment_matrix = np.zeros([similarity_matrix.shape[0], similarity_matrix.shape[1]])
        for i in range(len(row_ind)):
            assignment_matrix[row_ind[i], col_ind[i]] = 1

        # For debugging, display similarity matrix and assignment matrix
        if similarity_matrix.shape[0] > 0:
            print('row_ind: ', row_ind)
            print('col_ind: ', col_ind)
            cv2.imshow('similarity', cv2.resize(similarity_matrix, (600, 600), interpolation=cv2.INTER_NEAREST))
            cv2.imshow('assignment', cv2.resize(assignment_matrix, (600, 600), interpolation=cv2.INTER_NEAREST))
        return assignment_matrix

    def add_tracklet(self, tracklet):
        """
        Add a tracklet to the active tracklets after giving it a new ID.
        :param tracklet: The tracklet to be added.
        """
        tracklet.id = self.max_id
        self.max_id += 1
        self.tracklets_active.append(tracklet)

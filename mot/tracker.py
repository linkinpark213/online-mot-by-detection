import cv2
import numpy as np


class Tracker:
    def __init__(self, detector, matcher):
        self.detector = detector
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
        # Detection
        boxes = self.detector(img)

        # Data Association
        row_ind, col_ind, features = self.matcher(self.tracklets_active, boxes, img)

        # Tracklet Update
        self.update(row_ind, col_ind, boxes, features)

        self.frame_num += 1

    def update(self, row_ind, col_ind, detection_boxes, detection_features):
        """
        Update the tracklets.
        :param row_ind: A list of integers. Indices of the matched tracklets.
        :param col_ind: A list of integers. Indices of the matched detections.
        :param detection_boxes: A list of numpy arrays with shape (n, 5). Detected boxes.
        :param detection_features: The features of the detections. It can be any form you want.
        """
        raise NotImplementedError('Extend the Tracker class to implement your own updating strategy.')

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

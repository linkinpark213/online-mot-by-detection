import cv2
import mot
import mot.associate
import mot.detect
import mot.metric
import mot.encode
import numpy as np
import scipy.optimize
from mot.tracklet import Tracklet


class Tracker:
    def __init__(self, detector, encoder, metric, matcher):
        self.detector = detector
        self.encoder = encoder
        self.metric = metric
        self.matcher = matcher
        self.tracklets = []

    def tick(self, img):
        # Detection
        boxes = self.detector(img)

        # Feature Extraction
        features = self.encoder(boxes, img)

        # Affinity Calculation
        similarity_matrix = self.calculate_similarities(self.tracklets, features)

        # Data Association
        row_ind, col_ind = self.matcher(similarity_matrix)
        assignment_matrix = np.zeros([len(self.tracklets), len(features)])
        for i in range(len(row_ind)):
            assignment_matrix[row_ind[i], col_ind[i]] = 1

        # For debugging, display similarity matrix and assignment matrix
        if (similarity_matrix.shape[0] > 0):
            cv2.imshow('similarity', cv2.resize(similarity_matrix, (600, 600), interpolation=cv2.INTER_NEAREST))
            cv2.imshow('assignment', cv2.resize(assignment_matrix, (600, 600), interpolation=cv2.INTER_NEAREST))

        # Tracklets Update
        self.tracklets = []
        for feature in features:
            self.tracklets.append(Tracklet(feature))
        return assignment_matrix

    def calculate_similarities(self, tracklets, features):
        matrix = np.zeros([len(tracklets), len(features)])
        for i in range(len(tracklets)):
            for j in range(len(features)):
                matrix[i][j] = self.metric(tracklets[i].feature, features[j])
        return matrix


if __name__ == '__main__':
    detector = mot.detect.CenterNetDetector(conf_threshold=0.3)
    encoder = mot.encode.BoxCoordinateEncoder()
    metric = mot.metric.IoUMetric()
    matcher = mot.associate.HungarianMatcher()

    tracker = Tracker(detector, encoder, metric, matcher)

    capture = cv2.VideoCapture('/home/linkinpark213/Dataset/DukeMTMC/videos/camera8/00001.MTS')
    while True:
        ret, image = capture.read()
        if not ret:
            break
        boxes = tracker.tick(image)
        for tracklet in tracker.tracklets:
            box = tracklet.feature
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=1)
        cv2.imshow('Video', image)
        key = cv2.waitKey()
        if key == 27:
            break

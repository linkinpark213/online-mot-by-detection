import cv2
import mot
import mot.associate
import mot.detect
import mot.metric
import mot.encode


class Tracker:
    def __init__(self, detector, encoder, metric, matcher):
        self.detector = detector
        self.encoder = encoder
        self.metric = metric
        self.matcher = matcher

    def tick(self, img):
        boxes = self.detector(img)
        features = self.encoder(boxes, img)
        return features
        # distance_matrix = self.metric(features)
        # association_matrix = self.matcher(distance_matrix)
        # return association_matrix


if __name__ == '__main__':
    detector = mot.detect.CenterNetDetector(conf_threshold=0.4)
    encoder = mot.encode.BoxCoordinateEncoder()
    metric = mot.metric.IoUMetric()
    matcher = mot.associate.HungarianMatcher()

    tracker = Tracker(detector, encoder, metric, matcher)

    capture = cv2.VideoCapture('/home/linkinpark213/Dataset/DukeMTMC/videos/camera8/00000.MTS')
    while True:
        ret, image = capture.read()
        if not ret:
            break
        boxes = tracker.tick(image)
        for box in boxes:
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=1)
        cv2.imshow('Video', image)
        key = cv2.waitKey(30)
        if key == 27:
            break

import numpy as np
from mot.detect.detect import Detector, Detection


class MOTPublicDetector(Detector):
    def __init__(self, det_file_path, conf_threshold=0):
        super(MOTPublicDetector).__init__()
        self.frame_id = 0
        self.detections = np.loadtxt(det_file_path, delimiter=',')
        self.detections = self.detections[:, [0, 2, 3, 4, 5, 6]]
        self.detections[:, 3] = self.detections[:, 3] + self.detections[:, 1]
        self.detections[:, 4] = self.detections[:, 4] + self.detections[:, 2]
        self.detections = self.detections[np.where(self.detections[:, 5] > conf_threshold)]

    def __call__(self, img):
        self.frame_id += 1
        boxes = self.detections[np.where(self.detections[:, 0] == self.frame_id)][:, 1:]
        return [Detection(boxes[i][0:4], boxes[4]) for i in range(len(boxes))]

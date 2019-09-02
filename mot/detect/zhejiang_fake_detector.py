import numpy as np
from .detector import Detector


class ZhejiangFakeDetector(Detector):
    def __init__(self, path, conf_threshold=0.5, height_threshold=100, hw_ratio_lower_bound=1, hw_ratio_upper_bound=5):
        super(ZhejiangFakeDetector).__init__()
        self.conf_threshold = conf_threshold
        self.detections = np.loadtxt(path, delimiter=",")
        mask = self.detections[:, 4] / self.detections[:, 3] > hw_ratio_lower_bound
        mask = np.logical_and(mask, self.detections[:, 4] / self.detections[:, 3] < hw_ratio_upper_bound)
        mask = np.logical_and(mask, self.detections[:, 5] > self.conf_threshold)
        mask = np.logical_and(mask, self.detections[:, 4] > height_threshold)
        indices = np.where(mask == 1)
        self.detections = self.detections[indices]
        self.detections[:, 1] = self.detections[:, 1] - self.detections[:, 3] / 2
        self.detections[:, 2] = self.detections[:, 2] - self.detections[:, 4] / 2
        self.detections[:, 3] = self.detections[:, 1] + self.detections[:, 3]
        self.detections[:, 4] = self.detections[:, 2] + self.detections[:, 4]
        self.frame_num = 0

    def __call__(self, img):
        self.frame_num += 1
        mask = np.where(self.detections[:, 0] == self.frame_num)
        detections = self.detections[mask]
        return detections[:, 1:6]

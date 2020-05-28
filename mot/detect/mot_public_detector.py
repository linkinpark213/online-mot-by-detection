import numpy as np
from typing import List

from mot.structures import Detection
from mot.detect import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register()
class MOTPublicDetector(Detector):
    def __init__(self, det_file_path: str, conf_threshold: float = 0.5, **kwargs):
        super(MOTPublicDetector).__init__()
        self.frame_id = 0

        self.detections = np.loadtxt(det_file_path, delimiter=',')
        self.detections = self.detections[:, [0, 2, 3, 4, 5, 6]]
        self.detections[:, 3] = self.detections[:, 3] + self.detections[:, 1]
        self.detections[:, 4] = self.detections[:, 4] + self.detections[:, 2]
        self.detections = self.detections.astype(np.float32)
        self.detections = self.detections[np.where(self.detections[:, 5] >= conf_threshold)]

    def detect(self, img: np.ndarray) -> List[Detection]:
        self.frame_id += 1
        boxes = self.detections[np.where(self.detections[:, 0] == self.frame_id)][:, 1:]
        return [Detection(boxes[i][0:4], score=boxes[i][4]) for i in range(len(boxes))]

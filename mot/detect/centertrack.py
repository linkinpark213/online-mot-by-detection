import os
import sys
import numpy as np
from typing import List

from mot.structures import Detection
from .detect import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register()
class CenterTrackDetector(Detector):
    def __init__(self, cfg):
        # Add CenterTrack `src` and `src/lib` path to system path
        self.CENTERTRACK_PATH = cfg.src_path
        assert os.path.isdir(self.CENTERTRACK_PATH)
        sys.path.insert(0, self.CENTERTRACK_PATH)
        sys.path.insert(0, os.path.join(self.CENTERTRACK_PATH, 'lib'))

        from detectors import Detector as CTDetector
        from opts import opts

        self.opt = opts().init()
        self.opt.conf_thresh = cfg.conf_threshold
        self.detector = CTDetector(self.opt)

    def detect(self, img: np.ndarray) -> List[Detection]:
        box_result = self.detector.run(img)['results']
        box_result = box_result[1]

        if (box_result[:, 4] > self.opt.vis_thresh).any():

            box_result = box_result[box_result[:, 4] > self.opt.conf_thresh, :]
            detections = []
            for i in range(len(box_result)):
                detection = Detection(box_result[i][:4], box_result[i][4], None)
                detections.append(detection)
            else:
                return []

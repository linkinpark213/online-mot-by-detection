import os
import sys
import numpy as np
from typing import List

from mot.structures import Detection
from .detect import Detector, DETECTOR_REGISTRY

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../third_party', 'CenterNet', 'src'))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../third_party', 'CenterNet', 'src',
                                'lib'))
from detectors.detector_factory import detector_factory
from opts import opts


@DETECTOR_REGISTRY.register()
class CenterNetDetector(Detector):

    def __init__(self, checkpoint: str, arch: str, conf_threshold: float = 0.5, **kwargs):
        super(Detector).__init__()
        # Add CenterNet `src` and `src/lib` path to system path
        self.opt = opts().init('{} --load_model {} --arch {}'.format('ctdet', checkpoint, arch).split(' '))
        self.opt.conf_thresh = conf_threshold
        self.detector = detector_factory[self.opt.task](self.opt)

    def detect(self, img: np.ndarray) -> List[Detection]:
        box_result = self.detector.run(img)['results']
        detections = []
        for class_id in box_result.keys():
            class_result = box_result[class_id]

            if (class_result[:, 4] > self.opt.vis_thresh).any():
                class_result = class_result[class_result[:, 4] > self.opt.conf_thresh, :]

                # CenterNet id uses 1-based indexing
                detections.extend(
                    [Detection(result[:4], result[4], class_id=class_id - 1, mask=None) for result in class_result]
                )

        return detections

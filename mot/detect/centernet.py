import os
import sys
import numpy as np
from typing import List

from mot.structures import Detection
from .detect import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register()
class CenterNetDetector(Detector):

    def __init__(self, cfg):
        super(Detector).__init__()
        # Add CenterNet `src` and `src/lib` path to system path
        self.CENTERNET_PATH = cfg.src_path
        assert os.path.isdir(self.CENTERNET_PATH)
        sys.path.insert(0, self.CENTERNET_PATH)
        sys.path.insert(0, os.path.join(self.CENTERNET_PATH, 'lib'))

        from detectors.detector_factory import detector_factory
        from opts import opts

        self.opt = opts().init('{} --load_model {} --arch {}'.format('ctdet', cfg.checkpoint, cfg.arch).split(' '))
        self.opt.conf_thresh = cfg.conf_threshold
        self.detector = detector_factory[self.opt.task](self.opt)

    def detect(self, img: np.ndarray) -> List[Detection]:
        box_result = self.detector.run(img)['results']
        box_result = box_result[1]

        if (box_result[:, 4] > self.opt.vis_thresh).any():

            box_result = box_result[box_result[:, 4] > self.opt.conf_thresh, :]

            return [Detection(box_result[i][:4], box_result[i][4], None) for i in range(len(box_result))]

        else:
            return []

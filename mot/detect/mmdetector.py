import torch
from .detector import Detector
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result


class MMDetector(Detector):
    def __init__(self, config, checkpoint, conf_threshold=0.5):
        super(MMDetector).__init__()
        self.conf_thres = conf_threshold
        self.model = init_detector(
            config,
            checkpoint,
            device=torch.device('cuda', 0))

    def __call__(self, img):
        raw_result = inference_detector(self.model, img)[0][0]
        result = raw_result[np.where(raw_result[:, 4] > self.conf_thres)]
        return result


class HTCDetector(MMDetector):
    def __init__(self, conf_threshold=0.5):
        super(HTCDetector).__init__(
            'mot/detect/mmdet/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py',
            'mot/detect/mmdet/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
            conf_threshold)

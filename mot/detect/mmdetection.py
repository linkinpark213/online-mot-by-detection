import torch
from .detect import Detector, Detection
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
        raw_result = inference_detector(self.model, img)
        if isinstance(raw_result, tuple):
            box_result = raw_result[0][0]
            mask_result = raw_result[1][0]
            indices = np.where(box_result[:, 4] > self.conf_thres)[0]
            box_result = box_result[indices]
            mask_result = [mask_result[i] for i in indices]
            return [Detection(box_result[i][:4], box_result[i][4], mask_result[i]) for i in range(len(box_result))]
        else:
            raw_result = raw_result[0]
            result = raw_result[np.where(raw_result[:, 4] > self.conf_thres)]
            return [Detection(line[:4], line[4]) for line in result]

import torch
import numpy as np
from typing import List
from pycocotools.mask import decode
from mmdet.apis import inference_detector, init_detector

from .detect import Detector, DETECTOR_REGISTRY
from mot.structures import Detection


@DETECTOR_REGISTRY.register()
class MMDetector(Detector):
    def __init__(self, config: str, checkpoint: str, conf_threshold: float = 0.5, **kwargs):
        super(MMDetector).__init__()
        self.conf_thres = conf_threshold
        self.model = init_detector(
            config,
            checkpoint,
            device=torch.device('cuda', 0))

    def detect(self, img: np.ndarray) -> List[Detection]:
        raw_result = inference_detector(self.model, img)
        if isinstance(raw_result, tuple):
            # Only boxes with class "person" are counted
            box_result = raw_result[0][0]
            mask_result = raw_result[1][0]
            indices = np.where(box_result[:, 4] > self.conf_thres)[0]
            box_result = box_result[indices]
            mask_result = [mask_result[i] for i in indices]
            if mask_result != []:
                # If mask result is empty, decode() will raise error
                # mask_result = decode(mask_result).astype(np.bool)
                return [Detection(box_result[i][:4],
                                  box_result[i][4],
                                  mask_result[i])
                        for i in range(len(box_result))]

            return [Detection(box_result[i][:4],
                              box_result[i][4],
                              None)
                    for i in range(len(box_result))]
        else:
            # Only boxes with class "person" are counted
            raw_result = raw_result[0]
            result = raw_result[np.where(raw_result[:, 4] > self.conf_thres)]
            return [Detection(line[:4], line[4]) for line in result]

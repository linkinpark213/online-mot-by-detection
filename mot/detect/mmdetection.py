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
        detections = []
        if isinstance(raw_result, tuple):
            for class_id in range(len(raw_result[0])):
                box_result = raw_result[0][class_id]
                mask_result = raw_result[1][class_id]
                valid_indices = np.where(box_result[:, 4] > self.conf_thres)[0]
                box_result = box_result[valid_indices]
                mask_result = [mask_result[i] for i in valid_indices]
                if mask_result != []:
                    # If mask result is empty, decode() will raise error
                    # mask_result = decode(mask_result).astype(np.bool)
                    detections.extend([Detection(box_result[i][:4],
                                                 box_result[i][4],
                                                 class_id=class_id,
                                                 mask=mask_result[i])
                                       for i in range(len(box_result))])

                detections.extend([Detection(box_result[i][:4],
                                             box_result[i][4],
                                             class_id=class_id,
                                             mask=None)
                                   for i in range(len(box_result))])
        else:
            # Only boxes with class "person" are counted
            for class_id in range(len(raw_result)):
                class_result = raw_result[class_id]
                class_result = class_result[np.where(class_result[:, 4] > self.conf_thres)]
                detections.extend([Detection(line[:4], line[4], class_id=class_id) for line in class_result])

        return detections

import torch
import centermask
import numpy as np
from typing import List
from centermask.config import get_cfg
from detectron2.engine import DefaultPredictor

from .detect import Detector, DETECTOR_REGISTRY
from mot.structures import Detection


@DETECTOR_REGISTRY.register()
class CenterMaskDetector(Detector):
    def __init__(self, config: str, checkpoint: str, conf_threshold: float = 0.5, **kwargs):
        super(CenterMaskDetector, self).__init__()
        detectron2_cfg = get_cfg()
        detectron2_cfg.merge_from_file(config)
        detectron2_cfg.MODEL.WEIGHTS = checkpoint
        detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        self.conf_threshold = conf_threshold
        self.predictor = DefaultPredictor(detectron2_cfg)

    def detect(self, img: np.ndarray) -> List[Detection]:
        raw_results = self.predictor(img)['instances']
        pred_boxes = raw_results.pred_boxes.tensor.detach().cpu().numpy()
        pred_classes = raw_results.pred_classes.detach().cpu().numpy()
        scores = raw_results.scores.detach().cpu().numpy()

        if (len(scores) != 0):
            boxes = pred_boxes[np.where(pred_classes == 0)]
            scores = scores[np.where(pred_classes == 0)]
            boxes = boxes[np.where(scores >= self.conf_threshold)]
            scores = scores[np.where(scores >= self.conf_threshold)]
            detections = [Detection(pred_boxes[i], scores[i]) for i in range(len(boxes))]
            if hasattr(raw_results, 'pred_masks'):
                pred_masks = raw_results.pred_masks.detach().cpu().numpy()
                for i, detection in enumerate(detections):
                    detection.mask = pred_masks[i]

            if hasattr(raw_results, 'pred_keypoints'):
                pred_keypoints = raw_results.pred_keypoints.detach().cpu().numpy()[np.where(pred_classes == 0)]
                for i, detection in enumerate(detections):
                    detection.keypoints = pred_keypoints[i]
            return detections
        else:
            return []

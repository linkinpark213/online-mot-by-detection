import torch
import detectron2
import numpy as np
from typing import List
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from mot.structures import Detection
from .detect import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register()
class Detectron(Detector):
    def __init__(self, config: str, checkpoint: str, conf_threshold: float = 0.5, **kwargs):
        super(Detectron).__init__()
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
            # Can't make score filtering independent, though
            valid_indices = np.where(scores >= self.conf_threshold)
            pred_boxes = pred_boxes[valid_indices]
            scores = scores[valid_indices]
            pred_classes = pred_classes[valid_indices]
            detections = [Detection(pred_boxes[i], scores[i], pred_classes[i]) for i in range(len(pred_boxes))]

            if hasattr(raw_results, 'pred_masks'):
                pred_masks = raw_results.pred_masks.detach().cpu().numpy()
                pred_masks = pred_masks[valid_indices]
                for i, detection in enumerate(detections):
                    detection.mask = pred_masks[i]

            if hasattr(raw_results, 'pred_keypoints'):
                pred_keypoints = raw_results.pred_keypoints.detach().cpu().numpy()
                pred_keypoints = pred_keypoints[valid_indices]
                for i, detection in enumerate(detections):
                    detection.keypoints = pred_keypoints[i]
            return detections
        else:
            return []

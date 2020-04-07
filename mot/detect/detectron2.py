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
        self.predictor = DefaultPredictor(detectron2_cfg)

    def detect(self, img: np.ndarray) -> List[Detection]:
        raw_results = self.predictor(img)['instances']
        pred_boxes = raw_results.pred_boxes.tensor.detach().cpu().numpy()
        pred_classes = raw_results.pred_classes.detach().cpu().numpy()
        scores = raw_results.scores.detach().cpu().numpy()

        if (len(scores) != 0):
            boxes = pred_boxes[np.where(pred_classes == 0)]
            scores = scores[np.where(pred_classes == 0)]
            detections = [Detection(pred_boxes[i], scores[i]) for i in range(len(boxes))]

            if hasattr(raw_results, 'pred_keypoints'):
                pred_keypoints = raw_results.pred_keypoints.detach().cpu().numpy()[np.where(pred_classes == 0)]
                for i, detection in enumerate(detections):
                    detection.keypoints = pred_keypoints[i]
            return detections
        else:
            return []

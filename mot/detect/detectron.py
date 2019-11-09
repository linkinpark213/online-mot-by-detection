import torch
import detectron2
import numpy as np
from .detect import Detector, Detection
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detectron(Detector):
    def __init__(self, config, checkpoint, conf_threshold=0.5):
        super(Detectron).__init__()
        cfg = get_cfg()
        cfg.merge_from_file(config)
        cfg.MODEL.WEIGHTS = checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
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

import os
import cv2
import time
import torch
import argparse
import numpy as np
from typing import List
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import mot.utils
from mot.detect import Detector
from mot.structures import Detection


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
            boxes = pred_boxes[np.where(scores >= self.conf_threshold)]
            classes = pred_classes[np.where(scores >= self.conf_threshold)]
            scores = scores[np.where(scores >= self.conf_threshold)]
            detections = [Detection(boxes[i], scores[i], class_id=classes[i]) for i in range(len(boxes))]
            return detections
        else:
            return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/bdd100k.py')
    parser.add_argument('seqs_path', default='0',
                        help='Directory of the directories of test sequences. Leave it blank to use webcam.')
    parser.add_argument('results_path', default='',
                        help='Directory of store the output tracking result files. Leave it blank to disable.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    print('Initiating detector')
    detector = Detectron(cfg.tracker.detector.config,
                         cfg.tracker.detector.checkpoint,
                         cfg.tracker.detector.conf_threshold)

    for seq in os.listdir(args.seqs_path):
        demo_path = os.path.join(args.seqs_path, seq)
        capture = mot.utils.get_capture(demo_path)

        save_result = os.path.join(args.results_path, seq + '.txt')
        result_file = open(save_result, 'w+')

        frame_id = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_id += 1
            detections = detector(frame)
            print('Frame #{} - {} object(s) detected'.format(frame_id, len(detections)))
            for detection in detections:
                x, y, w, h = detection.box
                result_file.write('{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1\n'.format(
                    frame_id,
                    detection.class_id,
                    x,
                    y,
                    w,
                    h,
                    detection.score
                ))

        result_file.close()

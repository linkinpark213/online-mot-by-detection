import numpy as np
from .detector import Detector
from .yolov3 import YOLOv3


class YOLOv3Detector(Detector):
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4):
        super(YOLOv3Detector).__init__()
        self.yolo3 = YOLOv3('mot/detect/yolov3/cfg/yolo_v3.cfg',
                            'mot/detect/yolov3/yolov3.weights',
                            'mot/detect/yolov3/cfg/coco.names',
                            is_xywh=False,
                            conf_thresh=conf_threshold,
                            nms_thresh=nms_threshold)
        self.class_names = self.yolo3.class_names

    def __call__(self, img):
        bbox, cls_conf, cls_ids = self.yolo3(img)
        if bbox is not None:
            mask = cls_ids == 0

            bbox = bbox[mask]
            cls_conf = cls_conf[mask]

            return np.concatenate((bbox, np.expand_dims(cls_conf, 1)), 1)
        else:
            return []

import os
import cv2
import sys
import numpy as np
from ctypes import *
import os.path as osp
from typing import List, Tuple

from .detect import Detector, DETECTOR_REGISTRY
from mot.structures import Detection


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


@DETECTOR_REGISTRY.register()
class YOLO(Detector):
    def __init__(self, libPath: str, configPath: str, weightPath: str, metaPath: str, conf_threshold: float = 0.5,
                 nms_threshold: float = 0.45, **kwargs):
        super().__init__()
        configPath = os.path.abspath(configPath)
        weightPath = os.path.abspath(weightPath)
        metaPath = os.path.abspath(metaPath)
        assert osp.exists(libPath), "Invalid darknet library path `" + os.path.abspath(libPath) + "`"
        assert osp.exists(configPath), "Invalid config path `" + os.path.abspath(configPath) + "`"
        assert osp.exists(weightPath), "Invalid weight path `" + os.path.abspath(weightPath) + "`"
        assert osp.exists(metaPath), "Invalid meta path `" + os.path.abspath(metaPath) + "`"

        sys.path.append(os.path.abspath(os.path.dirname(libPath)))
        pwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(libPath)))
        import darknet

        self.darknet = darknet
        self.netMain = self.darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
        self.metaMain = self.darknet.load_meta(metaPath.encode("ascii"))
        self.darknet_image = self.darknet.make_image(darknet.network_width(self.netMain),
                                                     darknet.network_height(self.netMain), 3)
        os.chdir(pwd)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = (self.darknet.network_width(self.netMain), self.darknet.network_height(self.netMain))

    def _rescale(self, box: np.ndarray, ori_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
        box[0] = box[0] * (dst_size[0] / ori_size[0])
        box[1] = box[1] * (dst_size[1] / ori_size[1])
        box[2] = box[2] * (dst_size[0] / ori_size[0])
        box[3] = box[3] * (dst_size[1] / ori_size[1])
        return box

    def detect(self, img: np.ndarray) -> List[Detection]:
        self.image_size = (img.shape[1], img.shape[0])

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.input_size, interpolation=cv2.INTER_LINEAR)

        self.darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        num = c_int(0)
        pnum = pointer(num)
        self.darknet.predict_image(self.netMain, self.darknet_image)
        letter_box = 0
        dets = self.darknet.get_network_boxes(self.netMain, self.darknet_image.w, self.darknet_image.h,
                                              self.conf_threshold, self.conf_threshold, None, 0, pnum, letter_box)
        num = pnum[0]
        if self.nms_threshold:
            self.darknet.do_nms_sort(dets, num, self.metaMain.classes, self.nms_threshold)

        detections = []
        for j in range(num):
            for i in range(self.metaMain.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    detections.append(Detection(box=self._rescale(np.array([b.x - b.w / 2,
                                                                            b.y - b.h / 2,
                                                                            b.x + b.w / 2,
                                                                            b.y + b.h / 2]),
                                                                  self.input_size,
                                                                  self.image_size),
                                                score=dets[j].prob[i],
                                                class_id=i))

        detections = sorted(detections, key=lambda x: -x.score)
        self.darknet.free_detections(dets, num)
        print('Number of detections = ', len(detections))
        for detection in detections:
            print('Detection: Class ID = {}, Score = {}, W = {}, H = {}'.format(detection.class_id,
                                                                                detection.score,
                                                                                detection.box[2] - detection.box[0],
                                                                                detection.box[3] - detection.box[1]))
        return detections

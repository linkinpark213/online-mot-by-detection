import os
import re
import cv2
import sys
import numpy as np
from ctypes import *
import os.path as osp
from typing import List

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
    def __init__(self, configPath: str, weightPath: str, metaPath: str, **kwargs):
        super().__init__()
        assert osp.exists(configPath), "Invalid config path `" + os.path.abspath(configPath) + "`"
        assert osp.exists(weightPath), "Invalid weight path `" + os.path.abspath(weightPath) + "`"
        assert osp.exists(metaPath), "Invalid meta path `" + os.path.abspath(metaPath) + "`"

        lib = CDLL(osp.join(osp.abspath(osp.dirname(__file__)), '../../third_party', 'darknet', './libdarknet.so'),
                   RTLD_GLOBAL)

        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        self._copy_image_from_bytes = lib.copy_image_from_bytes
        self._copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        self._predict = lib.network_predict_ptr
        self._predict.argtypes = [c_void_p, POINTER(c_float)]
        self._predict.restype = POINTER(c_float)

        self._make_image = lib.make_image
        self._make_image.argtypes = [c_int, c_int, c_int]
        self._make_image.restype = IMAGE

        self._get_network_boxes = lib.get_network_boxes
        self._get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int,
                                            POINTER(c_int),
                                            c_int]
        self._get_network_boxes.restype = POINTER(DETECTION)

        self._make_network_boxes = lib.make_network_boxes
        self._make_network_boxes.argtypes = [c_void_p]
        self._make_network_boxes.restype = POINTER(DETECTION)

        self._free_detections = lib.free_detections
        self._free_detections.argtypes = [POINTER(DETECTION), c_int]

        self._free_batch_detections = lib.free_batch_detections
        self._free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        self._free_ptrs = lib.free_ptrs
        self._free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self._network_predict = lib.network_predict_ptr
        self._network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self._reset_rnn = lib.reset_rnn
        self._reset_rnn.argtypes = [c_void_p]

        self._load_net = lib.load_network
        self._load_net.argtypes = [c_char_p, c_char_p, c_int]
        self._load_net.restype = c_void_p

        self._load_net_custom = lib.load_network_custom
        self._load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self._load_net_custom.restype = c_void_p

        self._do_nms_obj = lib.do_nms_obj
        self._do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self._do_nms_sort = lib.do_nms_sort
        self._do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self._free_image = lib.free_image
        self._free_image.argtypes = [IMAGE]

        self._letterbox_image = lib.letterbox_image
        self._letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self._letterbox_image.restype = IMAGE

        self._load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self._load_image = lib.load_image_color
        self._load_image.argtypes = [c_char_p, c_int, c_int]
        self._load_image.restype = IMAGE

        self._rgbgr_image = lib.rgbgr_image
        self._rgbgr_image.argtypes = [IMAGE]

        self._predict_image = lib.network_predict_image
        self._predict_image.argtypes = [c_void_p, IMAGE]
        self._predict_image.restype = POINTER(c_float)

        self._predict_image_letterbox = lib.network_predict_image_letterbox
        self._predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self._predict_image_letterbox.restype = POINTER(c_float)

        self._network_predict_batch = lib.network_predict_batch
        self._network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                                c_float, c_float, POINTER(c_int), c_int, c_int]
        self._network_predict_batch.restype = POINTER(DETNUMPAIR)

        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath

        self.netMain = self._load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
        print('Loading meta')
        self.metaMain = self._load_meta(metaPath.encode("ascii"))
        print('Meta Loaded')
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            if os.path.exists(osp.join(osp.abspath(osp.dirname(metaPath)), result)):
                with open(result) as namesFH:
                    namesList = namesFH.read().strip().split("\n")
                    self.altNames = [x.strip() for x in namesList]

        print('YOLO Loaded')
        self.darknet_image = self._make_image(lib.network_width(self.netMain),
                                              lib.network_height(self.netMain), 3)

    @staticmethod
    def _convertBack(x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def detect(self, img: np.ndarray) -> List[Detection]:
        print('Performing detection')
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (self.darknet.network_width(self.netMain),
                                    self.darknet.network_height(self.netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        self.darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        detections = self._detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        return [Detection(box=np.array(detection[2][0] - detection[2][2] / 2,
                                       detection[2][1] - detection[2][3] / 2,
                                       detection[2][0] + detection[2][2] / 2,
                                       detection[2][1] + detection[2][3] / 2),
                          score=detection[1],
                          class_id=detection[0]) for detection in detections]

    def _detect_image(self, net, meta, im, thresh=.5, hier_thresh=.5, nms=.45, debug=True):
        num = c_int(0)
        if debug: print("Assigned num")
        pnum = pointer(num)
        if debug: print("Assigned pnum")
        self._predict_image(net, im)
        letter_box = 0
        # predict_image_letterbox(net, im)
        # letter_box = 1
        if debug: print("did prediction")
        # dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
        dets = self._get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        if debug: print("Got dets")
        num = pnum[0]
        if debug: print("got zeroth index of pnum")
        if nms:
            self._do_nms_sort(dets, num, meta.classes, nms)
        if debug: print("did sort")
        res = []
        if debug: print("about to range")
        for j in range(num):
            if debug: print("Ranging on " + str(j) + " of " + str(num))
            if debug: print("Classes: " + str(meta), meta.classes, meta.names)
            for i in range(meta.classes):
                if debug: print("Class-ranging on " + str(i) + " of " + str(meta.classes) + "= " + str(dets[j].prob[i]))
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    if self.altNames is None:
                        nameTag = meta.names[i]
                    else:
                        nameTag = self.altNames[i]
                    if debug:
                        print("Got bbox", b)
                        print(nameTag)
                        print(dets[j].prob[i])
                        print((b.x, b.y, b.w, b.h))
                    res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        if debug: print("did range")
        res = sorted(res, key=lambda x: -x[1])
        if debug: print("did sort")
        self._free_detections(dets, num)
        if debug: print("freed detections")
        return res

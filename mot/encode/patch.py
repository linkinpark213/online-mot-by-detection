import cv2
import numpy as np
from typing import List, Tuple

from mot.utils import crop
from mot.structures import Detection
from .encode import Encoder, ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class ImagePatchEncoder(Encoder):
    def __init__(self, resize_to: Tuple[int, int], name: str = 'patch', **kwargs):
        super(ImagePatchEncoder, self).__init__()
        self.resize_to: Tuple[int, int] = resize_to
        self.name: str = name

    def encode(self, detections: List[Detection], img: np.ndarray):
        imgs = []
        for detection in detections:
            box = detection.box
            patch = crop(img, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, int(max(box[2] - box[0], box[3] - box[1])))
            patch = cv2.resize(patch, self.resize_to)
            imgs.append(patch)
        return imgs

import cv2
import numpy as np
from typing import List, Tuple

from .encode import Encoder
from mot.structures import Detection


class ImagePatchEncoder(Encoder):
    def __init__(self, cfg):
        super(ImagePatchEncoder, self).__init__()
        self.resize_to: Tuple[int, int] = cfg.resize_to
        self.name: str = cfg.name if hasattr(cfg, 'name') else 'patch'

    def encode(self, detections: List[Detection], img: np.ndarray):
        imgs = []
        for detection in detections:
            box = detection.box
            patch = self.crop(img, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, max(box[2] - box[0], box[3] - box[1]))
            patch = cv2.resize(patch, self.resize_to)
            imgs.append(patch)
        return imgs

    @staticmethod
    def crop(img: np.ndarray, x_c: float, y_c: float, window_size: int):
        x_base = 0
        y_base = 0
        padded_img = img
        half_window = int(window_size // 2)
        if x_c < half_window:
            padded_img = cv2.copyMakeBorder(img, 0, 0, half_window, 0, borderType=cv2.BORDER_REFLECT)
            x_base = half_window
        if x_c > img.shape[1] - half_window:
            padded_img = cv2.copyMakeBorder(img, 0, 0, 0, half_window, borderType=cv2.BORDER_REFLECT)
        if y_c < half_window:
            padded_img = cv2.copyMakeBorder(img, half_window, 0, 0, 0, borderType=cv2.BORDER_REFLECT)
            y_base = half_window
        if y_c > img.shape[0] - half_window:
            padded_img = cv2.copyMakeBorder(img, 0, half_window, 0, 0, borderType=cv2.BORDER_REFLECT)

        return padded_img[int(y_base + y_c - half_window): int(y_base + y_c + half_window),
               int(x_base + x_c - half_window): int(x_base + x_c + half_window), :]

import cv2
import numpy as np
from typing import Union, Tuple, List


def crop(img: np.ndarray, x_c: float, y_c: float, window_size: Union[int, Tuple, List], resize_to: tuple = None,
         borderType: int = cv2.BORDER_REPLICATE):
    x_base = 0
    y_base = 0
    padded_img = img

    if type(window_size) is int:
        half_width = int(window_size // 2)
        half_height = half_width
    elif type(window_size) is tuple or type(window_size) is list:
        half_width = int(window_size[0] // 2)
        half_height = int(window_size[1] // 2)
    else:
        raise AssertionError(
            'Expecting a parameter `window_size` with type int/tuple/list, but got {}'.format(type(window_size)))

    if resize_to is not None:
        assert type(resize_to) is tuple, 'Expecting a parameter `resize_to` with type tuple'
        half_width = max(half_width, int(half_height * (resize_to[0] / resize_to[1])))
        half_width = max(half_height, int(half_width * (resize_to[1] / resize_to[0])))

    if x_c < half_width:
        padded_img = cv2.copyMakeBorder(padded_img, 0, 0, half_width, 0, borderType=borderType)
        x_base = half_width
    if x_c > img.shape[1] - half_width:
        padded_img = cv2.copyMakeBorder(padded_img, 0, 0, 0, half_width, borderType=borderType)
    if y_c < half_height:
        padded_img = cv2.copyMakeBorder(padded_img, half_height, 0, 0, 0, borderType=borderType)
        y_base = half_height
    if y_c > img.shape[0] - half_height:
        padded_img = cv2.copyMakeBorder(padded_img, 0, half_height, 0, 0, borderType=borderType)

    patch = padded_img[int(y_base + y_c - half_height): int(y_base + y_c + half_height),
            int(x_base + x_c - half_width): int(x_base + x_c + half_width), :]

    if resize_to is not None:
        patch = cv2.resize(patch, resize_to)

    return patch

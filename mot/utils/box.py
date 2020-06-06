import numpy as np
from typing import Union


def iou(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate the IoU (Intersection over Union) between one box and N boxes.

    Args:
        a: An np.ndarray of 4 float numbers or a numpy array. The one box in (l, t, r, b) format.
        b: An np.ndarray of N lists or N np.ndarrays of 4 float numbers, or a 2D numpy array of shape (N, 4). The other boxes for IoU comparison.

    Returns:
        iou: A floating number or an 1D numpy array. The IoU between box `a` and box(es) `b`.
    """
    a = np.array(a)
    b = np.array(b)
    assert len(a.shape) == 1, 'Expected 1-dimensional input `a`'
    assert len(b.shape) == 1 or len(b.shape) == 2, 'Expected 1-dimensional or 2-dimensional input `b`'

    ret_dim = len(b.shape)
    if len(b.shape) == 1:
        b = np.expand_dims(b, axis=0)

    b1_x1, b1_y1, b1_x2, b1_y2 = a[0:4]
    b2_x1, b2_y1, b2_x2, b2_y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    x1 = np.maximum(b1_x1, b2_x1)
    y1 = np.maximum(b1_y1, b2_y1)
    x2 = np.minimum(b1_x2, b2_x2)
    y2 = np.minimum(b1_y2, b2_y2)
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-16)
    iou[intersection == 0] = 0

    return iou[0] if ret_dim == 1 else iou


def expand(box: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """
    Expand a box by a margin factor.

    Args:
        box: An np.ndarray of shaoe (4,) or (N, 4). The box(es) in (l, t, r, b) format.
        margin: A floating number indicating the ratio between expanded margin of each edge and the box'es w/h.

    Returns:
        An np.ndarray with the same shape as input box. The expanded box(es).
    """
    assert len(box.shape) == 1 or len(box.shape) == 2, 'Expected 1-dimensional or 2-dimensional input `box`'
    assert margin >= 0.0, 'Expected a margin value > 0'
    expanded = box.copy()
    if len(expanded.shape) == 2:
        expanded = expanded.T
    w, h = expanded[2] - expanded[0], expanded[3] - expanded[1]
    expanded[0] -= w * margin
    expanded[1] -= h * margin
    expanded[2] += w * margin
    expanded[3] += h * margin
    if len(expanded.shape) == 2:
        expanded = expanded.T
    return expanded


def xyxy2xywh(box):
    measurement = box.copy()[:4]
    measurement[0] = (box[0] + box[2]) / 2
    measurement[1] = (box[1] + box[3]) / 2
    measurement[2] = box[2] - box[0]
    measurement[3] = box[3] - box[1]
    return measurement


def xyxy2xyah(box):
    measurement = xyxy2xywh(box)
    measurement[2] = measurement[2] / measurement[3]
    return measurement


def xyah2xywh(box):
    measurement = box.copy()[:4]
    measurement[2] = measurement[2] * measurement[3]
    return measurement


def xyah2xyxy(box):
    measurement = xyah2xywh(box)
    return xywh2xyxy(measurement)


def xywh2xyxy(box):
    measurement = box.copy()[:4]
    measurement[0] = measurement[0] - measurement[2] / 2
    measurement[1] = measurement[1] - measurement[3] / 2
    measurement[2] = measurement[0] + measurement[2]
    measurement[3] = measurement[1] + measurement[3]
    return measurement


def xywh2xyah(box):
    measurement = box.copy()[:4]
    measurement[2] = measurement[2] / measurement[3]
    return measurement


def xyxy2xtwh(box):
    measurement = box.copy()[:4]
    measurement[0] = (measurement[0] + measurement[2]) / 2
    measurement[2] = (measurement[2] - measurement[0]) * 2
    measurement[3] = measurement[3] - measurement[1]
    return measurement


def xywh2xtwh(box):
    measurement = box.copy()[:4]
    measurement[1] = measurement[1] - measurement[3] / 2
    return measurement


def xyah2xtwh(box):
    measurement = xyah2xywh(box)
    return xywh2xtwh(measurement)


def xtwh2xyxy(box):
    measurement = box.copy()[:4]
    measurement[0] = measurement[0] - measurement[2] / 2
    measurement[2] = measurement[0] + measurement[2]
    measurement[3] = measurement[1] + measurement[3]
    return measurement


def xtwh2xywh(box):
    measurement = box.copy()[:4]
    measurement[1] = measurement[1] + measurement[3] / 2
    return measurement


def xtwh2xyah(box):
    measurement = xtwh2xywh(box)
    return xywh2xyah(measurement)

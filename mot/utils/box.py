import numpy as np


def iou(a, b):
    """
    Calculate the IoU (Intersection over Union) between one box and N boxes
    :param a: A list of 4 float numbers or a numpy array. The one box.
    :param b: A list of N lists or N np.ndarrays of 4 float numbers, or a 2D numpy array of shape (N, 4). The other boxes for IoU comparison.
    :return: The IoU between box `a` and box(es) `b`. If there is only one box in `b`, only one float number will be returned.
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

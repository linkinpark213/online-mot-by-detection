import numpy as np
from typing import Union


def mask_iou(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate the IoU (Intersection over Union) between one mask and N masks.

    Args:
        a: A binary np.ndarray with shape (H, W). The one mask.
        b: A binary np.ndarray with shape (H, W) or (N, H, W). The other mask(s) for IoU comparison.

    Returns:
        iou: A floating number or an 1D numpy array. A The IoU between mask `a` and mask(s) `b`.
    """
    a = np.array(a)
    b = np.array(b)
    assert a is not None, 'Mask is not available in tracklet features.'
    assert len(a.shape) == 2, 'Expected 2-dimensional input `a`'
    assert len(b.shape) == 2 or len(b.shape) == 3, 'Expected 2-dimensional or 3-dimensional input `b`'

    ret_dim = len(b.shape) - 1

    intersection = np.logical_and(a, b)
    union = np.logical_or(a, b)
    if ret_dim == 1:
        iou = np.sum(intersection) / np.sum(union)
    else:
        iou = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))
    return iou

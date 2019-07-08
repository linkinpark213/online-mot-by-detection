from __future__ import division
import torch


def intersection_over_union(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True) -> torch.Tensor:
    """
    Returns the IoU of two bounding boxes, or one box and a series of boxes.
    :param box1: 2D tensor, a series of bounding boxes.
    :param box2: 2D tensor, another series of bounding boxes.
    :param xywh: Boolean value, whether the boxes' format is (center_x, center_y, w, h). If not, it would be (x1, y1, x2, y2)
    :return: A tensor. The IoU value of all pairs of boxes.
    """
    if xywh:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Calculate intersection box coordinates and its area.
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)
    intersection = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

    # Calculate union area.
    area1 = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    area2 = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union = area1 + area2 - intersection

    iou = intersection / (union + 1e-16)

    return iou

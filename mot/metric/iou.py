import numpy as np
from .metric import Metric


class IoUMetric(Metric):
    """
    An affinity metric that only considers the IoU of tracklets' box and detected box.
    """

    def __init__(self):
        super(IoUMetric).__init__()
        self.name = 'box'

    def __call__(self, tracklets, detected_boxes, img):
        matrix = np.zeros([len(tracklets), len(detected_boxes)])
        for i in range(len(tracklets)):
            for j in range(len(detected_boxes)):
                matrix[i][j] = self.iou(tracklets[i].last_box, detected_boxes[j])
        return matrix, detected_boxes

    def iou(self, a, b):
        b1_x1, b1_y1, b1_x2, b1_y2 = a[0:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = b[0:4]

        x1 = max(b1_x1, b2_x1)
        y1 = max(b1_y1, b2_y1)
        x2 = min(b1_x2, b2_x2)
        y2 = min(b1_y2, b2_y2)
        intersection = max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0)
        area1 = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        area2 = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        union = area1 + area2 - intersection

        return intersection / (union + 1e-16)

from .metric import Metric


class IoUMetric(Metric):
    def __init__(self):
        super(IoUMetric).__init__()

    def __call__(self, a, b):
        b1_x1, b1_y1, b1_x2, b1_y2 = a[0:3]
        b2_x1, b2_y1, b2_x2, b2_y2 = b[0:3]

        x1 = max(b1_x1, b2_x1)
        y1 = max(b1_y1, b2_y1)
        x2 = min(b1_x2, b2_x2)
        y2 = min(b1_y2, b2_y2)
        intersection = max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0)
        area1 = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        area2 = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        union = area1 + area2 - intersection

        return intersection / (union + 1e-16)

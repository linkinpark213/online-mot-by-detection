import numpy as np
from .encode import Encoder


class BoxEncoder(Encoder):
    def __init__(self, format='xywh'):
        super(BoxEncoder).__init__()
        self.format = format
        self.name = 'box_' + format

    def __call__(self, boxes, img, tracklet=None):
        if self.format == 'xywh':
            xywh = np.array(boxes)
            xywh[:, 2] = xywh[:, 2] - xywh[:, 0]
            xywh[:, 3] = xywh[:, 3] - xywh[:, 1]
            return xywh
        else:
            return boxes

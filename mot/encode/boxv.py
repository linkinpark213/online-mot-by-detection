import numpy as np
from .encode import Encoder


class UVSRUVSEncoder(Encoder):
    def __init__(self):
        super(UVSRUVSEncoder).__init__()

    def __call__(self, boxes, img, tracklet=None):
        if (tracklet is None or tracklet.history is []):
            return np.vstack((
                np.array([(boxes[:, 0] + boxes[:, 2]) / 2]),
                np.array([(boxes[:, 1] + boxes[:, 3]) / 2]),
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]),
                np.zeros(len(boxes)),
                np.zeros(len(boxes)),
                np.zeros(len(boxes))
            ))
        else:
            return np.vstack((
                np.array([(boxes[:, 0] + boxes[:, 2]) / 2]),
                np.array([(boxes[:, 1] + boxes[:, 3]) / 2]),
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]),
                np.array([(boxes[:, 0] + boxes[:, 2]) / 2]) - tracklet.history[-1][0],
                np.array([(boxes[:, 1] + boxes[:, 3]) / 2]) - tracklet.history[-1][1],
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - tracklet.history[-1][2]
            ))

import logging
import numpy as np
import mot.utils.box
from .metric import Metric


class IoUMetric(Metric):
    """
    An affinity metric that only considers the IoU of tracklets' box and detected box.
    """

    def __init__(self, use_prediction=False):
        self.encoding = 'box'
        self.use_prediction = use_prediction
        super(IoUMetric, self).__init__('box', history=1)

    def __call__(self, tracklets, detection_features):
        matrix = np.zeros([len(tracklets), len(detection_features)])
        for i in range(len(tracklets)):
            for j in range(len(detection_features)):
                if self.use_prediction:
                    matrix[i][j] = mot.utils.box.iou(tracklets[i].prediction.box, detection_features[j][self.encoding])
                else:
                    matrix[i][j] = mot.utils.box.iou(tracklets[i].last_detection.box,
                                                     detection_features[j][self.encoding])

        self._log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

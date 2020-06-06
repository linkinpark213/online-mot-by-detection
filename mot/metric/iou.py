import numpy as np
import mot.utils.box
from typing import List, Dict, Union

from mot.structures import Tracklet
from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class IoUMetric(Metric):
    """
    An affinity metric that only considers the IoU of tracklets' box and detected box.
    """

    def __init__(self, encoding='box', use_prediction=False, expand_margin: float = 0.0, **kwargs):
        self.encoding = encoding
        self.use_prediction = use_prediction
        self.expand_margin = expand_margin
        super(IoUMetric, self).__init__(encoding, **kwargs)

    def affinity_matrix(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[
        np.ndarray, List[List[float]]]:
        if len(detection_features) == 0:
            return np.array([[-1]])
        matrix = np.zeros([len(tracklets), len(detection_features)])
        det_boxes = np.stack([feature[self.encoding] for feature in detection_features])
        for i in range(len(tracklets)):
            a = tracklets[i].prediction.box if self.use_prediction else tracklets[i].last_detection.box
            b = det_boxes
            if self.expand_margin != 0.0:
                a = mot.utils.box.expand(a, self.expand_margin)
                b = mot.utils.box.expand(b, self.expand_margin)

            matrix[i] = mot.utils.box.iou(a, b)

        self._log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def similarity(self, tracklet_encoding: np.ndarray, detection_encoding: np.ndarray) -> Union[float, np.ndarray]:
        # To make this class not abstract
        return 0.0

import numpy as np
from typing import List, Dict, Union

import mot.utils.mask
from mot.structures import Tracklet
from .metric import Metric, METRIC_REGISTRY


@METRIC_REGISTRY.register()
class MaskIoUMetric(Metric):
    """
    An affinity metric that only considers the IoU of tracklets' box and detected box.
    """

    def __init__(self, encoding='mask', **kwargs):
        self.encoding = encoding
        super(MaskIoUMetric, self).__init__(encoding, **kwargs)

    def affinity_matrix(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[
        np.ndarray, List[List[float]]]:
        matrix = np.zeros([len(tracklets), len(detection_features)])
        det_masks = np.stack([feature[self.encoding] for feature in detection_features])
        for i in range(len(tracklets)):
            matrix[i] = mot.utils.mask.mask_iou(tracklets[i].feature[self.encoding], det_masks)

        self._log_affinity_matrix(matrix, tracklets, self.encoding)
        return matrix

    def similarity(self, tracklet_encoding: np.ndarray, detection_encoding: np.ndarray) -> Union[float, np.ndarray]:
        # To make this class not abstract
        return 0.0

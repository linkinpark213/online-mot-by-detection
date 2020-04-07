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

    def __init__(self, encoding='box', use_prediction=False, **kwargs):
        self.encoding = encoding
        self.use_prediction = use_prediction
        super(IoUMetric, self).__init__(encoding, **kwargs)

    def affinity_matrix(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Union[
        np.ndarray, List[List[float]]]:
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

    def similarity(self, tracklet_feature: Dict, detection_feature: Dict) -> float:
        # To make this class not abstract
        return 0.0

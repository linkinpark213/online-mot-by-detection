from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Union

from mot.structures import Tracklet
from mot.utils import Registry, Config
from mot.metric import Metric, build_metric

__all__ = ['Matcher', 'MATCHER_REGISTRY', 'build_matcher']

MATCHER_REGISTRY = Registry('matchers')


class Matcher(metaclass=ABCMeta):
    def __init__(self, metric: Config = None, threshold: float = 0, **kwargs):
        if metric is not None:
            # In cascade matcher, no metric is needed
            self.metric = build_metric(metric)
        self.threshold = threshold

    def __call__(self, tracklets: List[Tracklet], detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        return self.data_association(tracklets, detection_features)

    @abstractmethod
    def data_association(self, tracklets: List, detection_features: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Perform online data association between tracklets and features of detections.

        Args:
            tracklets: A list of tracklet objects, the tracklets to match.
                May not be all active tracklets in cases of cascade matching.
            detection_features: A list of dictionaries, the features of new detections.

        Returns:
            row_ind: A list of integers, the indices of matched tracklets.
            col_ind: A list of integers, the indices of matched detections, in correspondence to `row_ind`.
        """
        pass


def build_matcher(cfg):
    return MATCHER_REGISTRY.get(cfg.type)(**(cfg.to_dict()))

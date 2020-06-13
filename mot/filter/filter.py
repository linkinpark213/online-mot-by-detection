from typing import List, Callable

from mot.utils import Registry
from mot.structures import Detection

__all__ = ['DetectionFilter', 'build_detection_filter']

DETECTION_FILTER_REGISTRY = Registry('detection_filters')


class DetectionFilter:
    def __init__(self, filtering: Callable[[Detection], bool], **kwargs) -> None:
        self.filtering = filtering

    def __call__(self, detections: List[Detection]) -> List[Detection]:
        return self.filter(detections)

    def filter(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detection objects.

        Args:
            detections: A list of N detection objects.

        Returns:
            A list of N' Detection objects. (N' <= N)
        """
        return [detection for detection in detections if self.filtering(detection)]


def build_detection_filter(cfg):
    assert hasattr(cfg, 'type') or hasattr(cfg,
                                           'filtering'), 'Expected `type` or `filtering` param in detection filter config'
    if hasattr(cfg, 'type'):
        return DETECTION_FILTER_REGISTRY.get(cfg.type)(**(cfg.to_dict()))
    else:
        return DetectionFilter(**(cfg.to_dict()))

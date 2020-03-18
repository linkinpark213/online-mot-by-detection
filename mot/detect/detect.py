import numpy as np
from typing import List
from abc import abstractmethod, ABCMeta

from mot.structures import Detection
from mot.utils import Registry

__all__ = ['Detector', 'DETECTOR_REGISTRY', 'build_detector']

DETECTOR_REGISTRY = Registry('detectors')


class Detector(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    def __call__(self, img: np.ndarray) -> List[Detection]:
        return self.detect(img)

    @abstractmethod
    def detect(self, img: np.ndarray) -> List[Detection]:
        """
        Detect all objects in an image.

        Args:
            img: A numpy array of shape (H, W, 3)

        Returns:
            A list of N Detection objects.
        """
        pass


def build_detector(cfg):
    return DETECTOR_REGISTRY.get(cfg.type)(cfg)

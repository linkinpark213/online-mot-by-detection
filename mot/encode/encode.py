import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod

from mot.utils import Registry
from mot.structures import Detection

__all__ = ['Encoder', 'ENCODER_REGISTRY', 'build_encoder']

ENCODER_REGISTRY = Registry('encoders')


class Encoder(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        self.name: str = name if name is not None else 'encoding'

    def __call__(self, detections: List[Detection], img: np.ndarray) -> List[object]:
        return self.encode(detections, img)

    @abstractmethod
    def encode(self, detections: List[Detection], img: np.ndarray) -> List[object]:
        pass


def build_encoder(cfg):
    return ENCODER_REGISTRY.get(cfg.type)(cfg)

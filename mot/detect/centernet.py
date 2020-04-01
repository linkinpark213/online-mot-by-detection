import numpy as np
from typing import List

from mot.structures import Detection
from .detect import Detector, DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register()
class CenterNetDetector(Detector):

    def __init__(self):
        pass

    def detect(self, img: np.ndarray) -> List[Detection]:
        pass

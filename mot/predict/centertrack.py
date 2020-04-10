import numpy as np
from typing import List

from mot.structures import Prediction, Tracklet
from .predict import Predictor, PREDICTOR_REGISTRY


@PREDICTOR_REGISTRY.register()
class CenterTrackPredictor(Predictor):
    def __init__(self):
        pass

    def initiate(self, tracklets: List[Tracklet]) -> None:
        pass

    def update(self, tracklets: List[Tracklet]) -> None:
        pass

    def predict(self, tracklets: List[Tracklet], img: np.ndarray) -> List[Prediction]:
        pass

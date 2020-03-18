import numpy as np


class Prediction:
    def __init__(self, box: np.ndarray, score: float, mask=None) -> None:
        self.box = box
        self.score = score

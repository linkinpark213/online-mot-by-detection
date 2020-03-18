import numpy as np


class Detection:
    def __init__(self, box: np.ndarray, score: float, mask=None) -> None:
        self.box = box
        self.score = score
        self.mask = mask

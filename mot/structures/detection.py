import numpy as np


class Detection:
    def __init__(self, box: np.ndarray, score: float, class_id: int = 0, mask=None) -> None:
        self.box = box
        self.score = score
        self.class_id = class_id
        self.mask = mask

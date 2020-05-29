import numpy as np


class Prediction:
    def __init__(self, box: np.ndarray, score: float, mask=None) -> None:
        # A numpy array with shape (4), namely, (x1, y1, x2, y2).
        self.box = box
        # A floating number indicating the confidence of the detection.
        self.score = score
        # A mask of any type. In our HTC implementation, it's a binary numpy array of shape (H, W).
        self.mask = mask

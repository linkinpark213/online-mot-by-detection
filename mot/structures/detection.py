import numpy as np


class Detection:
    def __init__(self, box: np.ndarray, score: float, class_id: int = 0, mask=None) -> None:
        # A numpy array with shape (4), namely, (x1, y1, x2, y2).
        self.box = box
        # A floating number indicating the confidence of the detection.
        self.score = score
        # An integer indicating the category.
        self.class_id = class_id
        # A mask of any type. In our HTC implementation, it's a binary numpy array of shape (H, W).
        self.mask = mask

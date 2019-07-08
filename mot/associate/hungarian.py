import numpy as np
from .matcher import Matcher


class HungarianMatcher(Matcher):
    def __init__(self):
        super(Matcher).__init__()

    def __call__(self, tracks, detections):
        return np.random.random([len(tracks), len(detections)])

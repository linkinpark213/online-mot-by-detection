from typing import List

from mot.utils.config import Config
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.associate import build_matcher
from mot.predict import build_predictor
from .tracker import Tracker, TRACKER_REGISTRY


@TRACKER_REGISTRY.register()
class TrackingByDetection(Tracker):
    def __init__(self, detector: Config, matcher: Config, encoders: List[Config], predictor: Config, **kwargs):
        # Allow detector to be None
        detector = build_detector(detector) if detector is not None else None
        matcher = build_matcher(matcher)
        encoders = [build_encoder(encoder_cfg) for encoder_cfg in encoders]
        predictor = build_predictor(predictor)
        super().__init__(detector, encoders, matcher, predictor, **kwargs)

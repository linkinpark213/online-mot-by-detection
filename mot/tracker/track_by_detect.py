from mot.detect import build_detector
from mot.associate import build_matcher
from mot.encode import build_encoder
from mot.predict import build_predictor
from .tracker import Tracker, TRACKER_REGISTRY


@TRACKER_REGISTRY.register()
class TrackingByDetection(Tracker):
    def __init__(self, cfg):
        detector = build_detector(cfg.detector)
        matcher = build_matcher(cfg.matcher)
        encoders = [build_encoder(encoder_cfg) for encoder_cfg in cfg.encoders]
        predictor = build_predictor(cfg.predictor)
        super().__init__(detector, encoders, matcher, predictor)

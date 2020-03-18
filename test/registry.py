from mot.detect import build_detector, DETECTOR_REGISTRY
from mot.encode import build_encoder, ENCODER_REGISTRY
from mot.metric import build_metric, METRIC_REGISTRY
from mot.associate import build_matcher, MATCHER_REGISTRY
from mot.tracker import build_tracker, TRACKER_REGISTRY


def test_registry():
    print(DETECTOR_REGISTRY.keys())
    print(ENCODER_REGISTRY.keys())
    print(METRIC_REGISTRY.keys())
    print(MATCHER_REGISTRY.keys())
    print(TRACKER_REGISTRY.keys())


def test_get_component():
    detector = build_detector(dict(type='MMDetector'))
    encoder = build_encoder(dict(type='DGNetEncoder'))
    metric = build_metric(dict(type='IoUMetric'))
    matcher = build_matcher(dict(type='HungarianMatcher'))
    tracker = build_tracker(dict(type='Tracktor'))


if __name__ == '__main__':
    test_registry()
    test_get_component()

from typing import List

from mot.utils import Config
from mot.encode import Encoder
from mot.detect import Detector
from mot.tracker import Tracker
from mot.associate import build_matcher
from mot.predict import build_predictor
from mot.filter import build_detection_filter


class LocalTracker(Tracker):
    def __init__(self, detector: Detector, encoders: List[Encoder], matcher: Config, predictor: Config, **kwargs):
        matcher = build_matcher(matcher)
        predictor = build_predictor(predictor)
        if 'detection_filters' in kwargs.keys():
            print('Building detection filters')
            kwargs['detection_filters'] = [build_detection_filter(filter_cfg) for filter_cfg in
                                           kwargs['detection_filters']]
        super().__init__(detector, encoders, matcher, predictor, **kwargs)

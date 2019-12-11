import mot.encode
import mot.metric
import numpy as np
import mot.associate
import mot.utils.box
from mot.tracklet import Tracklet
from examples.tracktor import CustomTracker as Tracktor


class CustomTracker(Tracktor):
    def __init__(self, sigma_active=0.5, lambda_active=0.6, lambda_new=0.3, max_ttl=30):
        super().__init__(sigma_active=sigma_active, lambda_active=lambda_active, lambda_new=lambda_new)
        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet')
        reid_metric = mot.metric.EuclideanMetric(reid_encoder.name, history=30)
        reid_metric = mot.metric.GatedMetric(reid_metric, 0.7)
        self.secondary_matcher = mot.associate.HungarianMatcher(reid_metric, 0.7)
        self.encoders.append(reid_encoder)
        self.max_ttl = max_ttl

from .filter import DetectionFilter, DETECTION_FILTER_REGISTRY


@DETECTION_FILTER_REGISTRY.register()
class HWRatioFilter(DetectionFilter):
    def __init__(self, hw_ratio_threshold: float, **kwargs):
        self.hw_ratio_threshold = hw_ratio_threshold
        self.filtering = lambda x: (x.box[3] - x.box[1]) / (x.box[2] - x.box[0]) > self.hw_ratio_threshold
        super().__init__(self.filtering)

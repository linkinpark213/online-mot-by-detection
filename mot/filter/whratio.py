from typing import Callable

from .filter import DetectionFilter, DETECTION_FILTER_REGISTRY


@DETECTION_FILTER_REGISTRY.register()
class WHRatioFilter(DetectionFilter):
    def __init__(self, filtering: Callable[[float], bool], **kwargs):
        self.ratio_filtering = filtering
        self.filtering = lambda x: self.ratio_filtering((x.box[2] - x.box[0]) / (x.box[3] - x.box[1]))
        super().__init__(self.filtering)

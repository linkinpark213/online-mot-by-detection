from typing import List

from .filter import DetectionFilter, DETECTION_FILTER_REGISTRY


@DETECTION_FILTER_REGISTRY.register()
class ClassIDFilter(DetectionFilter):
    def __init__(self, class_ids: List[int], **kwargs):
        """
        Filtering detections by class IDs.

        Args:
            class_ids: A tuple of int values. The class ids to be accepted.
        """
        self.class_ids = class_ids
        self.filtering = lambda x: x.class_id in self.class_ids
        super().__init__(self.filtering)

import torchreid
import numpy as np
from typing import List, Tuple
from torchreid.utils import FeatureExtractor

from mot.structures import Detection
from .encode import Encoder, ENCODER_REGISTRY

__all__ = ['TorchreidEncoder']


@ENCODER_REGISTRY.register()
class TorchreidEncoder(Encoder):
    def __init__(self, model_name: str, checkpoint_path: str, img_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=checkpoint_path,
            device='cuda',
            image_size=(img_size[1], img_size[0])
        )

    def encode(self, detections: List[Detection], full_img: np.ndarray) -> List[object]:
        if len(detections) > 0:
            features = []
            full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
            for detection in detections:
                box = detection.box
                crop = full_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                # Torchreid will do resizing
                features.append(self.extractor(crop))
            return features
        else:
            return []

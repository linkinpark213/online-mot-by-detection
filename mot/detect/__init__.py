from importlib import util as imputil

from .detect import Detector, DETECTOR_REGISTRY, build_detector

if imputil.find_spec('detectron2'):
    from .detectron2 import Detectron
    if imputil.find_spec('centermask'):
        from .centermask2 import CenterMaskDetector
if imputil.find_spec('mmdet'):
    from .mmdetection import MMDetector
from .centernet import CenterNetDetector
from .centertrack import CenterTrackDetector
from .mot_public_detector import MOTPublicDetector

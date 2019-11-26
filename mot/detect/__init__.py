import importlib

from .detect import Detector, Detection

if importlib.util.find_spec('detectron2'):
    from .detectron2 import Detectron
if importlib.util.find_spec('mmdet'):
    from .mmdetection import MMDetector
from .mot_public_detector import MOTPublicDetector

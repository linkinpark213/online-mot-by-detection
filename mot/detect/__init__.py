import importlib

from .detector import Detector

if importlib.util.find_spec('detectron2'):
    from .detectron import Detectron
if importlib.util.find_spec('mmdet'):
    from .mmdetector import MMDetector, HTCDetector
from .mot_public_detector import MOTPublicDetector

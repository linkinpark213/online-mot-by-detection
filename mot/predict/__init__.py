import importlib

from .predict import Predictor, PREDICTOR_REGISTRY, build_predictor
from .kalman import KalmanPredictor

if importlib.util.find_spec('detectron2'):
    from .detectron2 import DetectronRCNNPredictor
if importlib.util.find_spec('mmdet'):
    from .mmdetection import MMTwoStagePredictor
from .pysot import SiamRPNPredictor

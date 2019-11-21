import importlib

from .predict import Predictor, Prediction
from .kalman import KalmanPredictor

if importlib.util.find_spec('detectron2'):
    from .detectron import DetectronRCNNPredictor

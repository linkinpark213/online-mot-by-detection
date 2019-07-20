from .detector import Detector
from .centernet.opts import opts
from .centernet.detectors.ctdet import CtdetDetector


class CenterNetDetector(Detector):
    def __init__(self, conf_threshold=0.5):
        super(CenterNetDetector).__init__()

        opt = opts()
        self.detector = CtdetDetector(opt)
        self.conf_threshold = conf_threshold

    def __call__(self, img):
        temp = self.detector.run(img)['results'][1]
        temp = temp[temp[:, 4] > self.conf_threshold]
        return temp

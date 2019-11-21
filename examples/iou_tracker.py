import mot.detect
import mot.metric
import mot.associate
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self, sigma_conf=0.3):
        detector = mot.detect.Detectron(
            'https://raw.githubusercontent.com/facebookresearch/detectron2/22e04d1432363be727797a081e3e9d48981f5189/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl'
        )
        metric = mot.metric.IoUMetric()
        matcher = mot.associate.GreedyMatcher(metric, sigma=0.3)
        super().__init__(detector, [], matcher)
        self.sigma_conf = sigma_conf

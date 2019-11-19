import mot.detect
import mot.encode
import mot.metric
import mot.associate
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self, sigma_conf):
        detector = mot.detect.Detectron(
            '/home/linkinpark213/Source/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl')
        encoder = mot.encode.PCBEncoder('mot/encode/PCB/model/')
        metric = mot.metric.EuclideanMetric('pcb')
        matcher = mot.associate.GreedyMatcher(metric, sigma=0.7)
        super().__init__(detector, encoder, matcher)
        self.sigma_conf = sigma_conf

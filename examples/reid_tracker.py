import mot.detect
import mot.encode
import mot.metric
import mot.associate
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self, sigma_conf=0.8):
        detector = mot.detect.Detectron(
            '/home/linkinpark213/Source/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl')

        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet/')
        reid_metric = mot.metric.CosineMetric(reid_encoder.name)
        matcher = mot.associate.HungarianMatcher(reid_metric, sigma=sigma_conf)
        super().__init__(detector, [reid_encoder], matcher)

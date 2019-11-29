import mot.detect
import mot.encode
import mot.metric
import mot.associate
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self, sigma_conf=0.8):
        detector = mot.detect.MMDetector(
            '/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
            'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth'
        )
        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet/')
        reid_metric = mot.metric.CosineMetric(reid_encoder.name)
        matcher = mot.associate.HungarianMatcher(reid_metric, sigma=sigma_conf)
        super().__init__(detector, [reid_encoder], matcher)

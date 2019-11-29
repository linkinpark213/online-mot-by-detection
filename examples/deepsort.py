import mot.associate
import mot.encode
import mot.metric
import mot.detect
import mot.predict
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self):
        detector = mot.detect.MMDetector(
            '/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
            'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth'
        )
        iou_metric = mot.metric.IoUMetric()
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.3)

        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet/')

        reid_metric = mot.metric.CosineMetric(reid_encoder.name)
        reid_metric = mot.metric.GatedMetric(reid_metric, 0.9)
        combined_metric = mot.metric.ProductMetric((reid_metric, iou_metric))
        combined_matcher = mot.associate.HungarianMatcher(combined_metric, sigma=0.5)

        matcher = mot.associate.CascadeMatcher((combined_matcher, iou_matcher))
        predictor = mot.predict.KalmanPredictor(box_type='xyxy', predict_type='xywh')
        super().__init__(detector, [reid_encoder], matcher, predictor)

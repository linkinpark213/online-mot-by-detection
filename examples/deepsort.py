import mot.associate
import mot.encode
import mot.metric
import mot.detect
import mot.predict
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self):
        detector = mot.detect.Detectron(
            'https://raw.githubusercontent.com/facebookresearch/detectron2/22e04d1432363be727797a081e3e9d48981f5189/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',
            'detectron2://COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl'
        )
        iou_metric = mot.metric.IoUMetric()
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.3)

        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet/')

        reid_metric = mot.metric.CosineMetric('dgnet')
        combined_metric = mot.metric.ProductMetric((reid_metric, iou_metric))
        combined_matcher = mot.associate.HungarianMatcher(combined_metric, sigma=0.5)

        matcher = mot.associate.CascadeMatcher((combined_matcher, iou_matcher))
        predictor = mot.predict.KalmanPredictor(box_type='xyxy', predict_type='xywh')
        super().__init__(detector, [reid_encoder], matcher, predictor)

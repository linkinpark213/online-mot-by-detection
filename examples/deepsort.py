import mot.associate
import mot.encode
import mot.metric
import mot.detect
import mot.predict
from mot.tracker import Tracker


class CustomTracker(Tracker):
    def __init__(self):
        detector = mot.detect.Detectron(
            '/home/linkinpark213/Source/detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',
            'detectron2://COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl')
        iou_metric = mot.metric.IoUMetric()
        iou_matcher = mot.associate.HungarianMatcher(iou_metric, sigma=0.3)

        reid_encoder = mot.encode.DGNetEncoder('mot/encode/DGNet/outputs/checkpoints/')

        reid_metric = mot.metric.CosineMetric('dgnet')
        combined_metric = mot.metric.ProductMetric((reid_metric, iou_metric))
        combined_matcher = mot.associate.HungarianMatcher(combined_metric, sigma=0.5)

        matcher = mot.associate.CascadeMatcher((combined_matcher, iou_matcher))
        # self.predictor = mot.predict.KalmanPredictor(box_type='xyxy', predict_type='xywh')
        self.predictor = None
        super().__init__(detector, [reid_encoder], matcher)

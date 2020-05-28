tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        type='MMDetector',
        config='/home/linkinpark213/Source/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco.py',
        checkpoint='https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_x101_64x4d_fpn_1x_coco/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth',
        conf_threshold=0.5,
    ),
    encoders=[
        dict(
            type='DGNetEncoder',
            name='dgnet',
            model_path='mot/encode/DGNet/id_00100000.pt'
        )
    ],
    matcher=dict(
        type='CascadeMatcher',
        matchers=[
            dict(
                type='HungarianMatcher',
                metric=dict(
                    type='GatedMetric',
                    metric=dict(
                        type='CosineMetric',
                        encoding='dgnet',
                        history=1,
                        history_fusing=max,
                    ),
                    threshold=0.9,
                ),
            ),
            dict(
                type='HungarianMatcher',
                metric=dict(
                    type='IoUMetric'
                ),
                threshold=0.3,
            )],
    ),
    predictor=dict(
        type='KalmanPredictor',
        box_type='xyxy',
        predict_type='xywh',
        weight_position=1. / 40,
        weight_velocity=1. / 160
    ),
    max_ttl=5,
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = False
draw_skeletons = False
draw_association = False

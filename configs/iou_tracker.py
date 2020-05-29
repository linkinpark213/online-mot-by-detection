tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detector/mmdetection/faster_rcnn_x101_64x4d_fpn_1x_coco.py'
    ),
    encoders=[],
    matcher=dict(
        type='GreedyMatcher',
        metric=dict(
            type='IoUMetric'
        ),
        threshold=0.3
    ),
    predictor=None,
)

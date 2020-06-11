tracker = dict(
    type='Tracktor',
    detector=dict(
        include='./detect/mmdetection/faster_rcnn_x101_64x4d_fpn_1x_coco.py'
    ),
    encoders=[
        dict(
            type='ImagePatchEncoder',
            resize_to=(48, 96),
        ),
    ],
    matcher=dict(
        type='HungarianMatcher',
        metric=dict(
            type='IoUMetric',
            use_prediction=True,
        ),
        threshold=0.5,
    ),
    predictor=dict(
        include='./predict/mmdetection/faster_rcnn_x101_64x4d_fpn_1x.py'
    ),
)
sigma_active = 0.5
lambda_active = 0.6
lambda_new = 0.3

max_ttl = 100

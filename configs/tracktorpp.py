tracker = dict(
    type='Tracktor',
    detector=dict(
        include='./detect/mmdetection/faster_rcnn_x101_64x4d_fpn_1x_coco.py'
    ),
    encoders=[
        dict(
            include='./encode/dgnet.py'
        )
    ],
    matcher=dict(
        type='HungarianMatcher',
        metric=dict(
            type='IoUMetric',
            use_prediction=True,
        ),
        threshold=0.5,
    ),
    secondary_matcher=dict(
        type='HungarianMatcher',
        metric=dict(
            type='GatedMetric',
            metric=dict(
                type='EuclideanMetric',
                encoding='dgnet',
                history=30,
            ),
            threshold=0.7,
        ),
        threshold=0.7,
    ),
    predictor=dict(
        include='./predict/mmdetection/faster_rcnn_x101_64x4d_fpn_1x.py'
    ),
)
sigma_active = 0.5
lambda_active = 0.6
lambda_new = 0.3

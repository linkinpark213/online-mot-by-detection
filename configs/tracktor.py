tracker = dict(
    type='Tracktor',
    detector=dict(
        type='MMDetector',
        config='/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
        checkpoint='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth',
        conf_threshold=0.5,
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
        type='MMTwoStagePredictor',
        config='/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
        checkpoint='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth',
        conf_threshold=0.5,
    ),
)
sigma_active = 0.5
lambda_active = 0.6
lambda_new = 0.3

max_ttl = 100

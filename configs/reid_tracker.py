type = 'TrackingByDetection'
detector = dict(
    type='MMDetector',
    config='/home/linkinpark213/Source/mmdetection/configs/faster_rcnn_x101_64x4d_fpn_1x.py',
    checkpoint='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth',
    conf_threshold=0.5,
)
encoders = [
    dict(
        type='DGNetEncoder',
        name='dgnet',
        model_path='mot/encode/DGNet/'
    ),
]
matcher = dict(
    type='HungarianMatcher',
    metric=dict(
        type='CosineMetric',
        encoding='dgnet',
    ),
    threshold=0.8
)
predictor = None

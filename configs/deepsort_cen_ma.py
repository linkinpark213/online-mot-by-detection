tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        type='CenterMaskDetector',
        config='/home/linkinpark213/Source/centermask2/configs/centermask/centermask_V_99_eSE_FPN_ms_3x.yaml',
        checkpoint='/home/linkinpark213/Source/centermask2/models/centermask2-V-99-eSE-FPN-ms-3x.pth',
        conf_threshold=0.6,
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
                    name='gated',
                    metric=dict(
                        type='CosineMetric',
                        encoding='dgnet',
                        history=1,
                        history_fusing=max,
                    ),
                    threshold=0.8,
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
        weight_velocity=1. / 320
    ),
    max_ttl=30,
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = True
draw_skeletons = False
draw_association = False

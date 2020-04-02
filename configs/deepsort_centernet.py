tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        type='CenterNetDetector',
        arch='dla_34',
        src_path='/home/linkinpark213/Source/CenterNet/src',
        checkpoint='/home/linkinpark213/Source/CenterNet/models/ctdet_coco_dla_2x.pth',
        conf_threshold=0.5,
    ),
    encoders=[
        dict(
            type='DGNetEncoder',
            name='dgnet',
            model_path='mot/encode/DGNet/'
        )
    ],
    matcher=dict(
        type='CascadeMatcher',
        matchers=[
            dict(
                type='HungarianMatcher',
                metric=dict(
                    type='ProductMetric',
                    metrics=[
                        dict(
                            type='GatedMetric',
                            metric=dict(
                                type='CosineMetric',
                                encoding='dgnet',
                                history=30,
                            ),
                            threshold=0.9,
                        ),
                        dict(
                            type='IoUMetric',
                            use_prediction=False,
                        ),
                    ]
                ),
                threshold=0.3,
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
    ),
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = False
draw_skeletons = False
draw_association = False


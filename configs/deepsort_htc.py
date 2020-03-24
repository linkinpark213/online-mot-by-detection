tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        type='MMDetector',
        config='/home/linkinpark213/Source/mmdetection/configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py',
        checkpoint='https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
        conf_threshold=0.3,
    ),
    encoders=[
        dict(
            type='DGNetEncoder',
            name='dgnet',
            model_path='mot/encode/DGNet/',
        ),
        dict(
            type='ImagePatchEncoder',
            resize_to=(48, 96),
        ),
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
                            threshold=0.1,
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
draw_masks = True
draw_skeletons = False
draw_association = True

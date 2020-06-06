tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detector/detectron2/bdd100k_faster_rcnn_x_101_32x8d_fpn_2x_crop.py'
        # include='./detector/mmdetection/faster_rcnn_r50_fpn_2x_coco.py'
    ),
    encoders=[
        dict(
            include='./encoder/dgnet.py'
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
                                history_fusing=max,
                            ),
                            threshold=0.9,
                        ),
                        dict(
                            type='IoUMetric',
                            use_prediction=False,
                            expand_margin=0.2,
                            threshold=0.3,
                        ),
                    ]
                ),
                threshold=0.6,
            ),
            dict(
                type='HungarianMatcher',
                metric=dict(
                    type='CosineMetric',
                    encoding='dgnet',
                    history=30,
                    history_fusing=max,
                ),
                threshold=0.85,
            ),
        ],
    ),
    predictor=dict(
        type='KalmanPredictor',
        box_type='xyxy',
        predict_type='xtwh',
        weight_position=1. / 40,
        weight_velocity=1. / 160
    ),
    max_ttl=60,
    keep_finished_tracks=True,
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = False
draw_skeletons = False
draw_association = False

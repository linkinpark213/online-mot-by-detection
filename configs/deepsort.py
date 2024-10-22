tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detect/detectron2/bdd100k_faster_rcnn_x_101_32x8d_fpn_2x_crop.py'
        # include='./detector/mmdetection/faster_rcnn_r50_fpn_2x_coco.py'
    ),
    encoders=[
        dict(
            include='./encode/dgnet.py'
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
                                history_fusing=lambda x: sum(x) / len(x),
                            ),
                            threshold=0.9,
                        ),
                        dict(
                            type='IoUMetric',
                            use_prediction=True,
                            expand_margin=0.0,
                            threshold=0.3,
                        ),
                    ]
                ),
                threshold=0.6,
            ),
            dict(
                type='GreedyMatcher',
                metric=dict(
                    type='IoUMetric',
                    use_prediction=True,
                    expand_margin=0.0,
                    threshold=0.6,
                ),
            )
        ],
    ),
    predictor=dict(
        include='./predict/kalman/kalman_xtwh.py'
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

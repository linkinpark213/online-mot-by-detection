tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detect/detectron2/bdd100k_faster_rcnn_x_101_32x8d_fpn_2x_crop.py',
        # include='./detector/mmdetection/faster_rcnn_r50_fpn_2x_coco.py',
        conf_threshold=0.8,
        hw_ratio_threshold=1.5,
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
                type='GreedyMatcher',
                metric=dict(
                    type='IoUMetric',
                    use_prediction=True,
                    threshold=0.8,
                ),
                confident=False,
            ),
            dict(
                type='HungarianMatcher',
                metric=dict(
                    type='CosineMetric',
                    encoding='dgnet',
                    history=120,
                    history_fusing=max,
                ),
                threshold=0.92,
            ),
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
        ],
    ),
    predictor=dict(
        include='./predict/kalman/kalman_xtwh.py'
    ),
    max_ttl=120,
    min_time_lived=10,
    keep_finished_tracks=True,
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = False
draw_skeletons = False
draw_association = False

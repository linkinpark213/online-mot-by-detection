tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detect/centermask/centermask_V_99_sSE_FPN_ms_3x.py'
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
        include='./predict/kalman/kalman_xtwh.py'
    ),
    max_ttl=30,
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = True
draw_skeletons = False
draw_association = False
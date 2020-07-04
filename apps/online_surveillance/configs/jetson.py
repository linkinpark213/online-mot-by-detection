tracker = dict(
    type='MultiThreadTracker',
    detector=dict(
        include='./detect/trt_ctdet_coco_dlav0_1x.py',
        conf_threshold=0.8,
    ),
    detection_filters=[
        dict(
            type='WHRatioFilter',
            filtering=lambda x: x < 1,
        )
    ],
    encoders=[
        dict(
            include='./encode/trt_openreid_r50.py'
        ),
    ],
    matcher=dict(
        type='GreedyMatcher',
        metric=dict(
            type='GatedMetric',
            metric=dict(
                type='CosineMetric',
                encoding='openreid',
                history=10,
                history_fusing=lambda x: sum(x) / len(x),
            ),
            threshold=0.6,
        ),
    ),
    predictor=None,
    max_ttl=120,
    min_time_lived=10,
    keep_finished_tracks=True,
    central_address='163.221.68.100:5558',
)
draw_frame_num = True
draw_current_time = True

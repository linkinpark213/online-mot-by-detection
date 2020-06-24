tracker = dict(
    type='MultiThreadTracker',
    detector=dict(
        include='../configs/trt_ctdet_coco_dlav0_1x.py',
        conf_threshold=0.9,
        hw_ratio_threshold=2,
    ),
    encoders=[
        dict(
            include='./trt_openreid_r50.py'
        ),
    ],
    matcher=dict(
        type='GreedyMatcher',
        metric=dict(
            type='GatedMetric',
            metric=dict(
                type='CosineMetric',
                encoding='openreid',
                history=60,
                history_fusing=lambda x: sum(x) / len(x),
            ),
            threshold=0.6,
        ),
    ),
    predictor=None,
    max_ttl=120,
    min_time_lived=10,
    keep_finished_tracks=True,
)
draw_frame_num = False
draw_current_time = True

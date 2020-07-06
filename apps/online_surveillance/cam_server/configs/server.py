tracker = dict(
    type='SingleThreadTracker',
    detector=dict(
        include='../../../../configs/detect/centernet/ctdet_coco_dlav0_1x.py',
        conf_threshold=0.9,
    ),
    detection_filters=[
        dict(
            type='ClassIDFilter',
            class_ids=(0,),
        ),
        dict(
            type='SizeFilter',
            filtering=lambda w, h: w > 64 and h > 128
        ),
        dict(
            type='WHRatioFilter',
            filtering=lambda x: x < 0.5,
        )
    ],
    encoders=[
        dict(
            include='../../../../configs/encode/openreid_r50.py'
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
            threshold=0.65,
        ),
        threshold=0.65
    ),
    predictor=None,
    max_ttl=120,
    min_time_lived=5,
    keep_finished_tracks=False,
    central_address='163.221.68.100:5558',
    max_feature_history = 120
)
draw_frame_num = True
draw_current_time = True

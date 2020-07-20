tracker = dict(
    type='MultiThreadTracker',
    detector=dict(
        include='../../../../configs/detect/darknet/yolov4-tiny.py',
        conf_threshold=0.7,
        nms_threshold=0.5,
    ),
    detection_filters=[
        dict(
            type='ClassIDFilter',
            class_ids=(0,),
        ),
        dict(
            type='SizeFilter',
            filtering=lambda w, h: w > 48 and h > 96
        ),
        dict(
            type='WHRatioFilter',
            filtering=lambda x: x < 0.5,
        )
    ],
    encoders=[
        dict(
            include='./encode/trt_openreid_r50.py'
        ),
        dict(
            include='../../../../configs/encode/patch.py'
        ),
    ],
    matcher=dict(
        type='GreedyMatcher',
        metric=dict(
            type='GatedMetric',
            metric=dict(
                type='CosineMetric',
                encoding='openreid',
                history=30,
                history_fusing=lambda x: sum(x) / len(x),
            ),
            threshold=0.65,
        ),
        threshold=0.65,
    ),
    predictor=None,
    max_ttl=120,
    min_time_lived=3,
    keep_finished_tracks=False,
    central_address='163.221.68.100:5558',
)
draw_frame_num = True
draw_current_time = True
draw_detections = True

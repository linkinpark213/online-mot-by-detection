tracker = dict(
    type='SingleThreadTracker',
    detector=dict(
        # include='../../../../configs/detect/centernet/ctdet_coco_dlav0_1x.py',
        include='../../../../configs/detect/darknet/yolov4-tiny.py',
        conf_threshold=0.5,
        nms_threshold=0.5,
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
        threshold=0.65
    ),
    predictor=None,
    max_ttl=30,
    min_time_lived=3,
    keep_finished_tracks=False,
    central_address='163.221.68.100:5558',
    max_feature_history = 60
)
draw_frame_num = True
draw_current_time = True
draw_detections = True

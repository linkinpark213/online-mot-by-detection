tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detect/centernet/ctdet_coco_dlav0_1x.py',
        # include='./detect/detectron2/bdd100k_faster_rcnn_x_101_32x8d_fpn_2x_crop.py',
        # include='./detect/mmdetection/faster_rcnn_x101_64x4d_fpn_1x_coco.py',
        conf_threshold=0.9,
        hw_ratio_threshold=2,
    ),
    encoders=[
        dict(
            include='./encode/openreid.py'
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
            threshold=0.75,
        ),
    ),
    predictor=None,
    max_ttl=120,
    min_time_lived=10,
    keep_finished_tracks=True,
)

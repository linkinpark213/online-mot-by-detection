tracker = dict(
    type='TrackingByDetection',
    detector=dict(
        include='./detector/mmdetection/faster_rcnn_x101_64x4d_fpn_1x_coco.py'
    ),
    encoders=[
        dict(
            include='./encode/dgnet.py'
        ),
        dict(
            type='ImagePatchEncoder',
            resize_to=(128, 128),
        ),
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
                            ),
                            threshold=0.9,
                        ),
                        dict(
                            type='IoUMetric',
                            use_prediction=False,
                        ),
                    ]
                ),
                threshold=0.3,
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
        type='SiamRPNPredictor',
        config='/home/linkinpark213/Source/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml',
        checkpoint='/home/linkinpark213/Source/pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth',
        x_size=(256, 256),
        max_batch_size=15,
    ),
)
confirmed_only = True
detected_only = True
draw_predictions = True
draw_masks = False
draw_skeletons = False
draw_association = False

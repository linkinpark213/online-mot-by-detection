import cv2
import mmcv
import torch
import random
import argparse
import numpy as np
from mmcv.runner import load_checkpoint
from mmaction.models import build_recognizer

import mot.utils
import mot.detect
import mot.encode
import mot.metric
import mot.associate
from mot.tracker import Tracker, build_tracker
from mot.utils.config import cfg_from_file, Config


class TSN:
    def __init__(self, config, checkpoint):
        cfg = mmcv.Config.fromfile(config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.data.test.test_mode = True

        if cfg.data.test.oversample == 'three_crop':
            cfg.model.spatial_temporal_module.spatial_size = 8

        self.model = build_recognizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(self.model, checkpoint, strict=True)

        self.model.eval()

    def __call__(self, images):
        images = np.array(images)
        images = images.transpose((0, 3, 1, 2))
        images = np.expand_dims(images, 0)
        images = images.astype(np.float32) - 128
        return self.model([1], None, return_loss=False, img_group_0=torch.Tensor(images))


def ramdom_sample(images, num_segments):
    total_images = len(images)
    image_inds = []
    segment_length = int(total_images / num_segments)
    for i in range(num_segments):
        image_inds.append(random.randint(segment_length * i, segment_length * i + segment_length - 1))
    return [images[ind] for ind in image_inds]


def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (int(width), int(height)))
    else:
        class MuteVideoWriter():
            def write(self, *args, **kwargs):
                pass

            def release(self):
                pass

        return MuteVideoWriter()


def track_and_recognize(tracker, recognizer, args):
    capture = cv2.VideoCapture(args.video_path)
    video_writer = get_video_writer(args.save_video, capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)
        frame = mot.utils.snapshot_from_tracker(tracker)

        # Perform action recognition each second
        for tracklet in tracker.tracklets_active:
            if tracklet.is_confirmed() and tracklet.is_detected() and len(tracklet.feature_history) >= 3:
                samples = ramdom_sample([feature[1]['patch'] for feature in tracklet.feature_history], 3)
                prediction = recognizer(samples)
                action = np.argmax(prediction[0])
                if action == 0:
                    box = tracklet.last_detection.box
                    frame = cv2.putText(frame, 'climbing', (int(box[0] + 4), int(box[1]) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

        cv2.imshow('Demo', frame)
        video_writer.write(frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/iou_tracker.py')
    parser.add_argument('recognizer_config', help='test config file of TSN action recognizer')
    parser.add_argument('recognizer_checkpoint', help='checkpoint file of TSN action recognizer')
    parser.add_argument('--video_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('--save_video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    args = parser.parse_args()

    cfg = cfg_from_file(args.tracker_config)
    cfg.tracker.encoders.append(Config(dict(
        type='ImagePatchEncoder',
        resize_to=(224, 224)
    )))
    kwargs = cfg.to_dict()

    tracker = build_tracker(cfg, **kwargs)

    recognizer = TSN(args.recognizer_config, args.recognizer_checkpoint)

    track_and_recognize(tracker, recognizer, args)

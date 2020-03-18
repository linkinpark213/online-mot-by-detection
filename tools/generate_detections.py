import os
import cv2
import time
import torch
import argparse
import numpy as np
from mmdet.apis import inference_detector, init_detector

from mot.utils.io import get_capture
from mot.utils.visualize import snapshot_from_detection


class HTCDetector:
    def __init__(self, args):
        self.model = init_detector(args.det_config, args.det_checkpoint, device=torch.device('cuda', args.device))

    def __call__(self, img):
        raw_result = inference_detector(self.model, img)[0][0]
        result = raw_result.copy()
        result = result[:, [0, 0, 0, 1, 2, 3, 4]]
        result[:, 0] = 0
        result[:, 1] = 0
        return result, np.zeros(len(raw_result))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos_path", type=str)
    # MMDetection arguments
    parser.add_argument('--det-config', help='detector test config file path',
                        default='configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
    parser.add_argument('--det-checkpoint', help='detector checkpoint file',
                        default='checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument('--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument('--score-thr', type=float, default=0.5, help='bbox score threshold')
    # FakeDetector arguments
    parser.add_argument('--det-save-path', help='path to detection files', default='results/det/')
    parser.add_argument('--frame-save-path', help='path to frame files', default='results/frames/')
    parser.add_argument("--save-frames", help='set true to save output frames of video', action='store_true',
                        default=False)
    return parser.parse_args()


def save_frame(img, save_path, frame):
    print('Frame saved to ' + save_path + '/' + 'frame{:05d}.jpg'.format(frame))
    return cv2.imwrite(save_path + '/' + 'frame{:05d}.jpg'.format(frame), img)


if __name__ == '__main__':
    args = parse_args()

    print('Initiating detector')
    detector = HTCDetector(args)

    if not os.path.isdir(args.det_save_path):
        os.makedirs(args.det_save_path)
    if not os.path.isdir(args.frame_save_path):
        os.makedirs(args.frame_save_path)

    for sequence in os.listdir(args.videos_path):
        sequence_name = sequence.split('.')[0]
        det_filename = os.path.join(args.det_save_path, '{}.txt'.format(sequence_name))
        args.demo_path = os.path.join(args.videos_path, sequence)

        # Video or image folder capture, same basic APIs
        capture = get_capture(args.demo_path)
        detector_output_file = open(det_filename, 'w+')
        frame_num = 0
        while True:
            ret, image = capture.read()
            if not ret:
                detector_output_file.close()
                break
            frame_num += 1

            print("Detecting Frame #{}".format(frame_num))
            bbox_ftxyxys, cls_ids = detector(image)

            # Save output frames (if required)
            if args.save_frames:
                if not os.path.exists(os.path.join(args.frame_save_path, sequence_name)):
                    os.makedirs(os.path.join(args.frame_save_path, sequence_name))

                img_write_path = os.path.join(args.frame_save_path, sequence_name, '{:06d}.jpg'.format(frame_num))
                image = snapshot_from_detection(image, bbox_ftxyxys)
                cv2.imwrite(img_write_path, image)
                print('Frame saved to ' + img_write_path)

            # Save output detection results
            for i in range(len(bbox_ftxyxys)):
                detector_output_file.write(
                    '{:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, 0\n'.format(frame_num,
                                                                               bbox_ftxyxys[i, 2],
                                                                               bbox_ftxyxys[i, 3],
                                                                               bbox_ftxyxys[i, 4],
                                                                               bbox_ftxyxys[i, 5],
                                                                               bbox_ftxyxys[i, 6]))

        detector_output_file.close()

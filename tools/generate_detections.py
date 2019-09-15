import os
import cv2
import time
import torch
import argparse
import numpy as np
from mmdet.apis import inference_detector, init_detector


class HTCDetector:
    def __init__(self, args):
        self.model = init_detector(args.det_config, args.det_checkpoint, device=torch.device('cuda', args.device))

    def __call__(self, img):
        raw_result = inference_detector(self.model, img)[0][0]
        result = raw_result.copy()
        result[:, 0] = (raw_result[:, 0] + raw_result[:, 2]) / 2
        result[:, 1] = (raw_result[:, 1] + raw_result[:, 3]) / 2
        result[:, 2] = raw_result[:, 2] - raw_result[:, 0]
        result[:, 3] = raw_result[:, 3] - raw_result[:, 1]
        return result[:, 0:4], result[:, 4], np.zeros(len(raw_result))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos_path", type=str)
    parser.add_argument("--result_path", type=str, default='results/')
    # MMDetection arguments
    parser.add_argument('--det_config', help='detector test config file path',
                        default='configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py')
    parser.add_argument('--det_checkpoint', help='detector checkpoint file',
                        default='checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    # FakeDetector arguments
    parser.add_argument('--det_save_path', help='path to detection files', default='results/det/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    detector = HTCDetector(args)

    args = parse_args()
    if not os.path.isdir(args.result_path):
        os.mkdir(args.result_path)

    for video_file in os.listdir(args.videos_path):
        sequence_name = video_file.split('.')[0]
        args.detection_save_path = os.path.join(args.det_save_path, '{}.txt'.format(sequence_name))
        args.video_or_images_path = os.path.join(args.videos_path, video_file)

        capture = cv2.VideoCapture(args.video_or_images_path)
        detector_output_file = open(args.detection_save_path, 'w+')
        frame_num = 0
        while True:
            ret, image = capture.read()
            if not ret:
                detector_output_file.close()
                break
            frame_num += 1
            bbox_xcycwh, cls_conf, cls_ids = detector(image)

            print("Detecting Frame #{}".format(frame_num))
            for i in range(len(bbox_xcycwh)):
                detector_output_file.write(
                    '{:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, 0\n'.format(frame_num,
                                                                               bbox_xcycwh[i, 0],
                                                                               bbox_xcycwh[i, 1],
                                                                               bbox_xcycwh[i, 2],
                                                                               bbox_xcycwh[i, 3],
                                                                               cls_conf[i]))

        detector_output_file.close()
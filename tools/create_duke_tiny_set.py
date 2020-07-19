import os
import shutil
import argparse
import numpy as np
import os.path as osp

VIDEO_START_TIMES = np.array([0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766])
DATA_START_FRAMES = np.array([0, 44158, 46094, 26078, 18913, 50742, 27476, 30733, 2935])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cams', type=str, help='Camera IDs separated with \',\'.')
    parser.add_argument('start_time', type=int,
                        help='Starting time (globally). Expected a value bigger than 53321.')
    parser.add_argument('n_frames', type=int, help='Number of frames to sample.')
    parser.add_argument('fps', type=int, help='Frames per second.')
    parser.add_argument('--images-path', type=str, required=True, help='Path to DukeMTMC/images.')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output images.')
    args = parser.parse_args()
    args.cams = list(map(int, args.cams.split(',')))

    if not osp.isdir(args.output_path):
        os.makedirs(args.output_path)

    START_FRAMES = args.start_time - VIDEO_START_TIMES

    for cam in args.cams:
        cam_output_path = osp.join(args.output_path, 'camera{}'.format(cam))
        if not osp.isdir(cam_output_path):
            os.makedirs(cam_output_path)

        for i in range(args.n_frames):
            frame_id = int(i * (60 / args.fps) + START_FRAMES[cam])
            original_path = osp.join(args.images_path, 'camera{:d}/{:06d}.jpg'.format(cam, frame_id))
            new_path = osp.join(cam_output_path, '{:06d}.jpg'.format(frame_id))
            shutil.copy(original_path, new_path)
            print('copying {} to {}'.format(original_path, new_path))

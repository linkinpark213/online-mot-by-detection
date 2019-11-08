import os
import cv2
import argparse
import importlib
import mot.utils.vis
from mot.utils import ImagesCapture


def run_demo(tracker, args):
    if args.demo_path == '':
        capture = cv2.VideoCapture(0)
    else:
        if os.path.isdir(args.demo_path):
            capture = ImagesCapture(args.demo_path)
        elif os.path.isfile(args.demo_path):
            capture = cv2.VideoCapture(args.demo_path)
        else:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')

    n = 0
    while True:
        ret, image = capture.read()
        if not ret:
            break
        n += 1
        if args.save_video != '' and n == 1:
            writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                     (image.shape[1], image.shape[0]))
        tracker.tick(image)
        image = mot.utils.vis.draw_tracklets(image, tracker.tracklets_active)
        if args.save_video != '':
            writer.write(image)
        else:
            cv2.imshow('Demo', image)
            key = cv2.waitKey(1)
            if key == 27:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='examples/deepsort_tracker.py')
    parser.add_argument('--demo_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('--save_video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    args = parser.parse_args()

    tracker = importlib.import_module(args.tracker_config)

    run_demo(tracker, args)

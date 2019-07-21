import os
import cv2
import argparse
import utils.vis
from utils.capture import ImagesCapture


def run_demo(tracker):
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    args = parser.parse_args()

    if args.demo_path == '':
        capture = cv2.VideoCapture(0)
    else:
        if os.path.isdir(args.demo_path):
            capture = ImagesCapture(args.demo_path)
        elif os.path.isfile(args.demo_path):
            capture = cv2.VideoCapture(args.demo_path)
        else:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')

    while True:
        ret, image = capture.read()
        if not ret:
            break
        tracker.tick(image)
        image = utils.vis.draw_tracklets(image, tracker.tracklets_active)
        cv2.imshow('Demo', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

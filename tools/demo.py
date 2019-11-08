import os
import cv2
import logging
import argparse
import mot.utils
import importlib.util


def run_demo(tracker, args):
    if args.demo_path == '':
        capture = cv2.VideoCapture(0)
    else:
        if os.path.isdir(args.demo_path):
            capture = mot.utils.ImagesCapture(args.demo_path)
        elif os.path.isfile(args.demo_path):
            capture = cv2.VideoCapture(args.demo_path)
        else:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')

    n = 0
    video_writer = None
    result_writer = None
    while True:
        ret, image = capture.read()
        if not ret:
            break
        n += 1
        if n == 1:
            if args.save_video != '':
                video_writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                               (image.shape[1], image.shape[0]))
            if args.save_result != '':
                result_writer = open(args.save_result, 'w+')
        tracker.tick(image)
        image = mot.utils.draw_tracklets(image, tracker.tracklets_active)
        image = mot.utils.draw_frame_num(image, n)

        # Write to video if demanded.
        if video_writer is not None:
            video_writer.write(image)

        # Write to result file if demanded.
        if result_writer is not None:
            result_writer.write(mot.utils.snapshot_to_mot(tracker))

        # Display image if demanded.
        if args.display:
            cv2.imshow('Demo', image)
            key = cv2.waitKey(1)
            if key == 27:
                break

    # Close writers after tracking
    if video_writer is not None:
        video_writer.release()
    if result_writer is not None:
        result_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='examples/deepsort_tracker.py')
    parser.add_argument('--demo_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('--save_video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save_result', default='', required=False,
                        help='Path to the output tracking result file. Leave it blank to disable.')
    parser.add_argument('--ignore_display', action='store_false', default=False, required=False, dest='display',
                        help='Add \'--ignore_display\' to only write to video / result file')
    parser.add_argument('--debug', action='store_true', default=False, required=False, dest='debug',
                        help='Add \'--debug\' to show lower-leveled loggings')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)

    # Load tracker from tracker definition script
    # See example trackers in the `example` folder
    spec = importlib.util.spec_from_file_location('CustomTracker', args.tracker_config)
    tracker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker_module)

    tracker = tracker_module.CustomTracker()

    run_demo(tracker, args)

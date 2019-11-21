import os
import cv2
import logging
import argparse
import mot.utils
import importlib.util


def get_capture(demo_path):
    if demo_path == '':
        return cv2.VideoCapture(0)
    else:
        if os.path.isdir(args.demo_path):
            return mot.utils.ImagesCapture(args.demo_path)
        elif os.path.isfile(args.demo_path):
            return cv2.VideoCapture(args.demo_path)
        else:
            raise AssertionError('Parameter "demo_path" is not a file or directory.')


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


def get_result_writer(save_result_path):
    if args.save_result != '':
        return open(save_result_path, 'w+')
    else:
        class MuteResultWriter():
            def write(self, *args, **kwargs):
                pass

            def close(self):
                pass

        return MuteResultWriter()


def run_demo(tracker, args):
    capture = get_capture(args.demo_path)
    video_writer = get_video_writer(args.save_video, capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result_writer = get_result_writer(args.save_result)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)
        image = mot.utils.visualize_snapshot(frame, tracker, draw_predictions=True, draw_skeletons=True)

        # Write to video if demanded.
        video_writer.write(image)

        # Write to result file if demanded.
        result_writer.write(mot.utils.snapshot_to_mot(tracker))

        # Display image if demanded.
        if args.display:
            cv2.imshow('Demo', image)
            key = cv2.waitKey(0 if args.debug else 1)
            if key == 27:
                break

    # Close writers after tracking
    video_writer.release()
    result_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='examples/deepsort.py')
    parser.add_argument('--demo_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('--save_video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save_result', default='', required=False,
                        help='Path to the output tracking result file. Leave it blank to disable.')
    parser.add_argument('--ignore_display', action='store_false', default=True, required=False, dest='display',
                        help='Add \'--ignore_display\' to only write to video / result file')
    parser.add_argument('--debug', action='store_true', default=False, required=False, dest='debug',
                        help='Add \'--debug\' to show lower-leveled loggings')
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load tracker from tracker definition script
    # See example trackers in the `example` folder
    spec = importlib.util.spec_from_file_location('CustomTracker', args.tracker_config)
    tracker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker_module)

    tracker = tracker_module.CustomTracker()

    run_demo(tracker, args)

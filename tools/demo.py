import cv2
import logging
import argparse
import mot.utils
import importlib.util


def run_demo(tracker, args):
    capture = mot.utils.get_capture(args.demo_path)
    video_writer = mot.utils.get_video_writer(args.save_video, capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                              capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result_writer = mot.utils.get_result_writer(args.save_result)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)
        image = mot.utils.visualize_snapshot(frame, tracker, draw_predictions=True, draw_skeletons=False)

        # Write to video if demanded.
        video_writer.write(image)

        # Write to result file if demanded.
        result_writer.write(mot.utils.snapshot_to_mot(tracker))

        # Display image if demanded.
        if args.display:
            cv2.imshow('Demo', image)
            key = cv2.waitKey(0 if hasattr(args, 'frame_by_frame') and args.frame_by_frame else 1)
            if key == 27:
                break
            elif key == 32:
                if not hasattr(args, 'frame_by_frame'):
                    args.frame_by_frame = True
                else:
                    args.frame_by_frame = not args.frame_by_frame

    tracker.terminate()

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

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('MOT')
    if args.debug:
        logger.setLevel(logging.DEBUG)
        if args.display:
            logger.__setattr__('display', True)
    else:
        logger.setLevel(logging.INFO)

    # Load tracker from tracker definition script
    # See example trackers in the `example` folder
    spec = importlib.util.spec_from_file_location('CustomTracker', args.tracker_config)
    tracker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker_module)

    tracker = tracker_module.CustomTracker()

    run_demo(tracker, args)

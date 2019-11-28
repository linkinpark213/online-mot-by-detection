import os
import cv2
import logging
import argparse
import importlib
from mot.utils import evaluate_mot_online

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='examples/deepsort.py')
    parser.add_argument('--mot_subset_path', required=True,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('--output_path', required=True,
                        help='Path to the output tracking result file.')
    parser.add_argument('--save_video', required=True,
                        help='Path to the output video file. Leave it blank to disable saving video.')
    parser.add_argument('--ignore_display', action='store_false', default=True, required=False, dest='display',
                        help='Add \'--ignore_display\' to only write to video / result file')
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

    evaluate_mot_online(tracker, args.mot_subset_path, output_path=args.output_path, output_video_path=args.save_video,
                        show_result=(not args.display))

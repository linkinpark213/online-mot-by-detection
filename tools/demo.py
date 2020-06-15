import cv2
import logging
import argparse

import mot.utils
from mot.tracker import build_tracker


def run_demo(tracker, args, **kwargs):
    capture = mot.utils.get_capture(args.demo_path)
    video_writer = None
    result_writer = mot.utils.get_result_writer(args.save_result)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)
        image = mot.utils.snapshot_from_tracker(tracker, **kwargs)

        # Write to video if demanded. Video size may change because of extra contents visualized.
        if tracker.frame_num == 1:
            video_writer = mot.utils.get_video_writer(args.save_video, image.shape[1], image.shape[0])

        video_writer.write(image)

        # Write to result file if demanded.
        result_writer.write(mot.utils.snapshot_to_mot(tracker, mots=args.mots))

        # Display image if demanded.
        if args.display:
            cv2.imshow('Demo', cv2.resize(image, (image.shape[1], image.shape[0])))
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
    if video_writer is not None:
        video_writer.release()
    result_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--demo-path', default='0', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save-result', default='', required=False,
                        help='Path to the output tracking result file. Leave it blank to disable.')
    parser.add_argument('--ignore-display', action='store_false', default=True, required=False, dest='display',
                        help='Add \'--ignore_display\' to only write to video / result file')
    parser.add_argument('--debug', action='store_true', default=False, required=False, dest='debug',
                        help='Add \'--debug\' to show lower-leveled loggings')
    parser.add_argument('--save-log', default='', required=False,
                        help='Path to save the logs. Leave it blank to disable.')
    parser.add_argument('--mots', action='store_true', default=False, required=False, dest='mots',
                        help='Add \'--mots\' to enable RLE segmentation in MOT results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('MOT')
    if args.debug:
        logger.setLevel(logging.DEBUG)
        if args.display:
            logger.__setattr__('display', True)
    else:
        logger.setLevel(logging.INFO)

    if args.save_log != '':
        handler = logging.FileHandler(args.save_log, mode='w+')
        logger.addHandler(handler)

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)

    run_demo(tracker, args, **kwargs)

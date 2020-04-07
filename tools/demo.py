import cv2
import logging
import argparse
import mot.utils

from mot.tracker import build_tracker
from mot.utils import cfg_from_file


def snapshot_to_mot(tracker, time_lived_threshold=1, ttl_threshold=3, detected_only=True):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and (
                tracklet.is_detected() or not detected_only):
            box = tracklet.last_detection.box
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                          tracklet.id,
                                                                                          box[0],
                                                                                          box[1],
                                                                                          box[2] - box[0],
                                                                                          box[3] - box[1])
    return data


def run_demo(tracker, args, **kwargs):
    capture = mot.utils.get_capture(args.demo_path)
    video_writer = None
    result_writer = mot.utils.get_result_writer(args.save_result)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)
        image = mot.utils.snapshot_from_tracker(frame, tracker, **kwargs)

        # Write to video if demanded. Video size may change because of extra contents visualized.
        if tracker.frame_num == 1:
            video_writer = mot.utils.get_video_writer(args.save_video, image.shape[1], image.shape[0])

        video_writer.write(image)

        # Write to result file if demanded.
        result_writer.write(snapshot_to_mot(tracker))

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
        handler = logging.FileHandler(args.save_log)
        logger.addHandler(handler)

    cfg = cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)

    run_demo(tracker, args, **kwargs)

import os
import cv2
import time
import logging
import datetime
import argparse
import numpy as np

import mot.utils
from mot.tracker import build_tracker

from importlib import util as imputil

if imputil.find_spec('tensorrt') and imputil.find_spec('torch2trt'):
    from encode import TRTOpenReIDEncoder
    from detect import TRTCenterNetDetector
else:
    logging.basicConfig(level=logging.DEBUG)
    logging.warning('TensorRT and torch2trt not found. TensorRT models unavailable.')

from tracker import MultiThreadTracker
from apps.online_surveillance.utils.io import SCTOutputWriter


def parse_args(parser):
    args = parser.parse_args()
    if args.save_result == '':
        args.save_result = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt'
    if args.save_video == '':
        args.save_video = 'videos/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.mp4'
    if args.start_time == 0:
        args.start_time = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    if tuple(args.resolution) != (0, 0):
        assert len(tuple(args.resolution)) == 2, 'Expected 2 integers as input resolution'
    return args


def simulate(tracker, args, **kwargs):
    start_time = time.mktime(time.strptime(args.start_time, '%Y%m%d-%H%M%S'))

    capture = mot.utils.get_capture(args.capture)
    capture = mot.utils.RealTimeCaptureWrapper(capture, original_fps=args.original_fps,
                                               start_time=start_time)

    writer = SCTOutputWriter(args, tracker.identifier, fps=args.original_fps)

    logging.getLogger('MOT').info('Writing tracking results to ' + str(args.save_result))
    logging.getLogger('MOT').info('Writing tracked camera video to ' + str(args.save_video))

    # Wait until synchronized start time
    if start_time > time.time():
        logging.getLogger('MOT-Simulator').info('Waiting for starting time {}. ({} secs to go)'.format(args.start_time,
                                                                                                       start_time - time.time()))
    while time.time() < start_time:
        pass

    # Start pipeline
    logging.getLogger('MOT-Simulator').info('Running real-time single-camera tracking simulation...')

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if tuple(args.resolution) != (0, 0):
                frame = cv2.resize(frame, tuple(args.resolution))
            tracker.tick(frame)

            image = mot.utils.snapshot_from_tracker(tracker, **kwargs)

            writer.write(tracker, frame, image)

        logging.getLogger('MOT').info('Capture ended. Terminating...')

    except KeyboardInterrupt:
        logging.getLogger('MOT').info('Keyboard interruption. Terminating...')
    finally:
        writer.close()
        capture.release()
        tracker.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', help='Path to tracker config file.')
    parser.add_argument('--capture', type=str, required=True,
                        help='Image/Video capture path for real-time tracking simulation.')
    parser.add_argument('--original-fps', type=int, required=False, default=60,
                        help='Frames per second of the image/video capture stream.')
    parser.add_argument('--resolution', type=int, nargs='+', required=False, default=(0, 0),
                        help='Expected resolution in (W, H)')
    parser.add_argument('--start-time', type=str, required=False, default=0,
                        help='Tracking start time for simulation. Format: YYYYmmDD-HHMMSS')
    parser.add_argument('--footage-port', type=int, default=5555, required=False,
                        help='TCP Port for streaming. Default: 5555.')
    parser.add_argument('--tracker-port', type=int, default=5556, required=False,
                        help='TCP Port for tracker state. Default: 5556.')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save-result', default='', required=False,
                        help='Path to the output tracking result file. Leave it blank to disable.')
    parser.add_argument('--save-log', default='', required=False,
                        help='Path to save the logs. Leave it blank to disable.')
    args = parse_args(parser)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s:%(name)s: %(asctime)s %(message)s')

    if args.save_log != '':
        save_log_dir = os.path.abspath(os.path.dirname(args.save_log))
        if not os.path.isdir(save_log_dir):
            logging.warning('Result saving path {} doens\'t exist. Creating...'.format(save_log_dir))
            os.makedirs(save_log_dir)

        handler = logging.FileHandler(args.save_log, mode='w+')
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(name)s: %(asctime)s %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger('MOT')
        logger.addHandler(handler)

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)

    simulate(tracker, args, **kwargs)

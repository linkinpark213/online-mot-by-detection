import cv2
import time
import logging
import datetime
import argparse

import mot.utils
from mot.tracker import build_tracker

from importlib import util as imputil

if imputil.find_spec('tensorrt') and imputil.find_spec('torch2trt'):
    from encode import TRTOpenReIDEncoder
    from detect import TRTCenterNetDetector
else:
    logging.warning('TensorRT and torch2trt not found. TensorRT models unavailable.')

from tracker import MultiThreadTracker
from apps.online_surveillance.utils.io import SCTOutputWriter


def parse_args(parser):
    args = parser.parse_args()
    if args.save_result == '':
        args.save_result = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt'
    if args.save_video == '':
        args.save_video = 'videos/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.mp4'
    return args


def run(tracker, args, **kwargs):
    capture = mot.utils.get_capture(args.capture)
    capture.set(cv2.CAP_PROP_FOCUS, args.focus_length)

    writer = SCTOutputWriter(args, tracker.identifier)

    logging.getLogger('MOT').info('Writing tracking results to ' + str(args.save_result))
    logging.getLogger('MOT').info('Writing raw camera video to ' + str(args.save_video)[:-4] + '_raw.mp4')
    logging.getLogger('MOT').info('Writing tracked camera video to ' + str(args.save_video))
    logging.getLogger('MOT').info('Running single-camera tracking...')

    while True:
        # To avoid network congestion and image data accumulation
        time.sleep(0.2)

        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)

        image = mot.utils.snapshot_from_tracker(tracker, **kwargs)

        writer.write(tracker, frame, image)

    tracker.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', help='Path to tracker config file.')
    parser.add_argument('--capture', type=str, default='1', required=False, help='Capture number or video file path.')
    parser.add_argument('--focus-length', type=int, default=100, required=False, help='Focus length.')
    parser.add_argument('--footage-port', type=int, default=5555, required=False,
                        help='TCP Port for streaming. Default: 5555.')
    parser.add_argument('--tracker-port', type=int, default=5556, required=False,
                        help='TCP Port for tracker state. Default: 5556.')
    parser.add_argument('--listen-port', type=int, default=5557, required=False,
                        help='TCP Port for listening multi-cam tracker\'s message')
    parser.add_argument('--central-address', type=str, default='163.221.68.100:5558', required=False,
                        help='IP:Port of central multi-cam re-identification server')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save-result', default='', required=False,
                        help='Path to the output tracking result file. Leave it blank to disable.')
    args = parse_args(parser)

    logging.basicConfig(level=logging.INFO)

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)

    run(tracker, args, **kwargs)

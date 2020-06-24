import cv2
import zmq
import base64
import logging
import argparse

import mot.utils
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.tracker import build_tracker

from trt_openreid import TRTOpenReIDEncoder
from trt_centernet import TRTCenterNetDetector
from mthread_tracker import MultiThreadTracker


def run(tracker, args, **kwargs):
    capture = cv2.VideoCapture(1)

    context = zmq.Context()
    footage_socket = context.socket(zmq.PUB)
    footage_socket.bind('tcp://*:' + str(args.port))

    print('Running single-camera tracking...')

    while True:

        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)

        image = mot.utils.snapshot_from_tracker(tracker, **kwargs)

        encoded, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)

    tracker.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', help='Path to tracker config file')
    parser.add_argument('--port', type=int, default=5555, required=False, help='TCP Port for streaming. Default: 5555')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save-result', default='', required=False,
                        help='Path to the output tracking result file. Leave it blank to disable.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)

    run(tracker, args, **kwargs)

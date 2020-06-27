import cv2
import zmq
import time
import base64
import logging
import datetime
import argparse

import mot.utils
from mot.encode import build_encoder
from mot.detect import build_detector
from mot.tracker import build_tracker

from trt_openreid import TRTOpenReIDEncoder
from trt_centernet import TRTCenterNetDetector
from mthread_tracker import MultiThreadTracker
from apps.online_surveillance.utils.format import snapshot_to_base64, image_to_base64


def run(tracker, args, **kwargs):
    capture = cv2.VideoCapture(1)
    video_writer = None
    if args.save_result == '':
        args.save_result = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.txt'
    result_writer = mot.utils.get_result_writer(args.save_result)
    logging.getLogger('MOT').info('Writing tracking results to ', args.save_result)

    # Streaming port
    context = zmq.Context()
    footage_socket = context.socket(zmq.PUB)
    footage_socket.bind('tcp://*:' + str(args.footage_port))
    # Tracker state port
    tracker_socket = context.socket(zmq.PUB)
    tracker_socket.bind('tcp://*:' + str(args.tracker_port))

    print('Running single-camera tracking...')

    while True:

        time.sleep(0.2)

        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)

        image = mot.utils.snapshot_from_tracker(tracker, **kwargs)

        # Save to video if necessary. Video size may change because of extra contents visualized.
        if tracker.frame_num == 1:
            video_writer = mot.utils.get_video_writer(args.save_video, image.shape[1], image.shape[0])
        video_writer.write(image)

        # Save to result file if necessary.
        result_writer.write(mot.utils.snapshot_to_mot(tracker))

        # Send to streaming port.
        footage_socket.send(image_to_base64(image))

        # Send tracker state to tracker port.
        tracker_socket.send(snapshot_to_base64(tracker))

    tracker.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', help='Path to tracker config file')
    parser.add_argument('--footage-port', type=int, default=5555, required=False,
                        help='TCP Port for streaming. Default: 5555')
    parser.add_argument('--tracker-port', type=int, default=5556, required=False,
                        help='TCP Port for tracker state. Default: 5556')
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

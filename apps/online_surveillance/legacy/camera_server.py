import os
import time
import base64
import logging
import argparse
from pymongo import MongoClient
from pymongo.database import Database

import mot.utils
from mot.utils import cfg_from_file
from mot.tracker import build_tracker
from mot.detect import MOTPublicDetector

from utils.format import snapshot_to_dicts


class DummyCameraServer:
    def __init__(self, tracker, capture, db: Database):
        self.tracker = tracker
        self.capture = capture
        self.db = db

    def run_forever(self):
        timestamp = time.time()
        self.db['events'].insert_one({
            'time': timestamp,
            'event': 'start'
        })

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    logging.error('Error fetching stream!')
                    break

                tracker.tick(frame)
                del tracker.tracklets_inactive
                del tracker.tracklets_finished
                tracker.tracklets_inactive = []
                tracker.tracklets_finished = []

                timestamp = time.time()
                points = snapshot_to_dicts(tracker, timestamp)
                self.db['detections'].insert_many(points)

                self.db['frames'].delete_many({})
                self.db['frames'].insert_one({
                    'time': timestamp,
                    'frame': str(base64.b64encode(frame))[2:-1],
                })
                logging.info('{} - Saving {} targets to database'.format(
                    time.strftime('%Y/%m/%D %H:%M:%S', time.localtime(timestamp)),
                    len(points) + 1)
                )
                del points
        finally:
            self.db['events'].insert_one({
                'time': timestamp,
                'event': 'stop'
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--input', type=str, help='Camera number or a path to the video file as dummy camera input')
    parser.add_argument('--dets', type=str, default='', help='MOT-format detection input, if available')
    args = parser.parse_args()

    cfg = cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)
    logging.basicConfig(level=logging.INFO)

    tracker = build_tracker(cfg.tracker)

    if os.path.isfile(args.dets):
        tracker.detector = MOTPublicDetector(args.dets, conf_threshold=-1)

    capture = mot.utils.get_capture(args.input)
    db_client = MongoClient("mongodb://localhost:27017/")

    server = DummyCameraServer(tracker, capture, db_client['motreid'])
    server.run_forever()

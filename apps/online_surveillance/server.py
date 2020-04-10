import base64
import logging
import argparse
import datetime
import mot.utils
from influxdb import InfluxDBClient

from mot.utils import cfg_from_file
from mot.tracker import build_tracker
from utils.format import snapshot_to_points


class CameraServer:
    def __init__(self, tracker, capture, db_client):
        self.tracker = tracker
        self.capture = capture
        self.db_client = db_client

    def run_forever(self):
        timestamp = datetime.datetime.utcnow().isoformat("T") + "Z"
        self.db_client.write_points([{
            'time': timestamp,
            'measurement': 'event',
            'tags': {
                'time': timestamp,
            },
            'fields': {
                'value': 'start'
            }
        }])

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    logging.error('Error fetching stream!')
                    break

                tracker.tick(frame)

                timestamp = datetime.datetime.utcnow().isoformat("T") + "Z"
                points = snapshot_to_points(tracker, timestamp)
                self.db_client.write_points(points)
                self.db_client.write_points([{
                    'time': timestamp,
                    'measurement': 'frame',
                    'tags': {
                        'time': timestamp,
                    },
                    'fields': {
                        'value': str(base64.b64encode(frame))[2:-1],
                    },
                }])
                logging.info(timestamp + ' Saving {} points'.format(len(points) + 1))
        finally:
            self.db_client.write_points([{
                'time': timestamp,
                'measurement': 'event',
                'tags': {
                    'time': timestamp,
                },
                'fields': {
                    'value': 'stop'
                }
            }])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--port', type=int, default=8086, help='InfluxDB port')
    args = parser.parse_args()

    cfg = cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)
    capture = mot.utils.get_capture('0')
    db_client = InfluxDBClient(host='localhost',
                               port=args.port,
                               username='root',
                               password='root',
                               database='motreid')

    server = CameraServer(tracker, capture, db_client)
    server.run_forever()

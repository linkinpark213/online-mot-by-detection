import base64
from typing import Dict, List

from mot.tracker import Tracker


def snapshot_to_points(tracker: Tracker, timestamp:str) -> List[Dict]:
    points = []
    for tracklet in tracker.tracklets_active:
        points.append({
            'time': timestamp,
            'measurement': 'box',
            'tags': {
                'id': str(tracklet.id),
            },
            'fields': {
                'l': tracklet.last_detection.box[0],
                't': tracklet.last_detection.box[1],
                'r': tracklet.last_detection.box[2],
                'b': tracklet.last_detection.box[3],
            },
        })
        points.append({
            'time': timestamp,
            'measurement': 'image',
            'tags': {
                'id': str(tracklet.id),
            },
            'fields': {
                'value': str(base64.b64encode(tracklet.feature['patch']))[2:-1],
            },
        })
        points.append({
            'time': timestamp,
            'measurement': 'feature',
            'tags': {
                'id': str(tracklet.id),
            },
            'fields': {
                'value': tracklet.feature['dgnet'],
            },
        })
    return points

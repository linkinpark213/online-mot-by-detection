import base64
from typing import Dict, List, Tuple

from mot.tracker import Tracker


def snapshot_to_dicts(tracker: Tracker, timestamp: float) -> Tuple[List[Dict], List[Dict]]:
    detections = []
    features = []
    for tracklet in tracker.tracklets_active:
        detections.append({
            'time': timestamp,
            'tracklet_id': tracklet.id,
            'box': {
                'l': float(tracklet.last_detection.box[0]),
                't': float(tracklet.last_detection.box[1]),
                'r': float(tracklet.last_detection.box[2]),
                'b': float(tracklet.last_detection.box[3]),
            },
        })
        features.append({
            'time': timestamp,
            'tracklet_id': tracklet.id,
            'image': str(base64.b64encode(tracklet.feature['patch']))[2:-1],
            'feature': tracklet.feature['dgnet'].tolist(),
        })
    return detections, features

import cv2
import base64
import numpy as np

from mot.tracker import Tracker


def snapshot_to_base64(tracker: Tracker) -> str:
    """
    Save current active and detected objects to a numpy array and convert to base64 text.

    Args:
        tracker: Single-camera multi-object tracker object.

    Returns:
        A string. The base64 encoding of a tracker state array with shape (N, 266).
            The second dimension is composed of 10 numbers (MOT-format data) + 256 (feature dimension).
            10 indicates (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>)
            We are keeping <x> <y> <z> though they are still useless by far.
            256 is the number of dimensions of OpenReID encoding.
    """
    data = []
    for tracklet in tracker.tracklets_active:
        if tracklet.is_detected():
            box = tracklet.last_detection.box
            score = tracklet.last_detection.score
            feature = tracklet.feature['openreid']
            detection = np.array([tracker.frame_num, tracklet.id, box[0], box[1], box[2], box[3], score, -1, -1, -1])
            feature = np.concatenate((detection, feature), axis=0)
            data.append(feature)

    data = np.stack(data, axis=0) if len(data) > 0 else np.zeros((0, 266))
    text = base64.b64encode(data)
    print('Length of base64 encoded snapshot: ', len(text))
    return text


def image_to_base64(image: np.ndarray) -> str:
    encoded, buffer = cv2.imencode('.jpg', image)
    text = base64.b64encode(buffer)
    print('Length of base64 encoded snapshot: ', len(text))
    return text

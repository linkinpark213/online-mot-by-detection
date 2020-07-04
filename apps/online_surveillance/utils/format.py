import cv2
import time
import base64
import numpy as np

from mot.tracker import Tracker


def snapshot_to_base64(tracker: Tracker, identifier: int) -> str:
    """
    Save current active and detected objects to a numpy array and convert to base64 text.

    Args:
        tracker: Single-camera multi-object tracker object.
        identifier: An integer as an identifier of the tracker.

    Returns:
        A bytearray. The base64 encoding of a tracker state array with shape (N, 266).
            The second dimension is composed of 1 (identifier) + 10 numbers (MOT-format data) + 256 (feature dimension).
            1 indicates a randomly generated integer that works as the identifier of the single-cam tracker.
            10 indicates (<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>).
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
    data = np.hstack((np.ones([len(data), 1]) * identifier, data))
    bytes = base64.b64encode(data)
    # print('Length of base64 encoded snapshot: ', len(bytes))
    return bytes


def image_to_base64(image: np.ndarray) -> str:
    encoded, buffer = cv2.imencode('.jpg', image)
    text = base64.b64encode(buffer)
    # print('Length of base64 encoded snapshot: ', len(text))
    return text


def snapshot_to_mot(tracker, time_lived_threshold=1, ttl_threshold=3, detected_only=True):
    data = ''
    current_time = time.time()
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and (
                tracklet.is_detected() or not detected_only):
            box = tracklet.last_detection.box
            score = tracklet.last_detection.score
            data += '{:f}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1\n'.format(current_time,
                                                                                              tracklet.id,
                                                                                              box[0],
                                                                                              box[1],
                                                                                              box[2] - box[0],
                                                                                              box[3] - box[1],
                                                                                              score)
    return data

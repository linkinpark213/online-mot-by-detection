import cv2
import numpy as np

_colors = [(47, 47, 211),
           (74, 195, 139),
           (34, 87, 255),
           (212, 188, 0),
           (176, 39, 156),
           (254, 109, 83),
           (136, 150, 0),
           (7, 193, 255),
           (139, 125, 96),
           (117, 117, 117),
           (72, 85, 121),
           (255, 138, 68),
           (45, 192, 251),
           (212, 188, 0),
           (57, 220, 205),
           (99, 30, 233),
           (97, 97, 97),
           (59, 235, 255),
           (80, 175, 76),
           (55, 64, 93),
           (43, 180, 175),
           (195, 244, 240),
           (54, 67, 244),
           (210, 205, 255),
           (162, 31, 123),
           (60, 142, 56),
           (201, 230, 200),
           (25, 74, 230),
           (188, 204, 255),
           (167, 151, 0),
           (189, 189, 189),
           (91, 24, 194),
           (251, 64, 224),
           (159, 63, 48),
           (233, 202, 197),
           (0, 124, 245),
           (107, 121, 0),
           (255, 77, 124),
           (33, 33, 33),
           (208, 187, 248)]

_keypoint_connections = [
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [0, 5],
    [0, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [5, 11],
    [6, 12],
    [11, 12],
    [11, 13],
    [12, 14],
    [13, 15],
    [14, 16]
]


def draw_tracklets(image, tracklets, confirmed_only=True, detected_only=True):
    """
    Draw the boxes of tracklets.
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param tracklets: A list of Tracklet objects. The currently active tracklets.
    :param confirmed_only: Set to True to draw boxes only for tracklets that are confirmed.
    :param detected_only: Set to True to draw boxes only for tracklets that are detected in the frame.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with boxes of tracklets drawn.
    """
    for tracklet in tracklets:
        if (tracklet.is_confirmed() or not confirmed_only) and (tracklet.is_detected() or not detected_only):
            image = draw_object(image, tracklet.last_detection.box, tracklet.id)
            if hasattr(tracklet.last_detection, 'keypoints'):
                image = draw_skeleton(image, tracklet.last_detection.keypoints)
    return image


def draw_frame_num(image, frame_num):
    """
    Draw the frame number at the top-left corner of the frame.
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param frame_num: Frame number.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with its frame number drawn.
    """
    cv2.putText(image, '{}'.format(frame_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
    return image


def draw_object(image, box, id):
    image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          _colors[int(id) % _colors.__len__()], thickness=3)
    image = cv2.putText(image, '{:d}'.format(int(id)), (int(box[0]), int(box[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, _colors[int(id) % _colors.__len__()], thickness=2)
    image = cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), radius=10,
                       color=(0, 0, 255), thickness=-1)
    return image


def draw_skeleton(image, keypoints, id):
    for connection in _keypoint_connections:
        image = cv2.line(image,
                         (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1])),
                         (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1])),
                         color=_colors[int(id) % _colors.__len__()], thickness=2)
    for keypoint in keypoints:
        image = cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 2, (255, 255, 255), thickness=-1)
    return image

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


def draw_targets(image, tracklets, confirmed_only=True, detected_only=True, draw_predictions=False,
                 draw_skeletons=True):
    """
    Draw the boxes of targets.
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param tracklets: A list of Tracklet objects. The currently active tracklets.
    :param confirmed_only: Set to True to draw boxes only for tracklets that are confirmed.
    :param detected_only: Set to True to draw boxes only for tracklets that are detected in the frame.
    :param draw_predictions: Set to True to draw prediction boxes for each target, if it's available.
    :param draw_skeletons: Set to True to draw skeletons of each target, if it's available.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with boxes of tracked targets drawn.
    """
    for tracklet in tracklets:
        if (tracklet.is_confirmed() or not confirmed_only) and (tracklet.is_detected() or not detected_only):
            if draw_predictions and tracklet.prediction is not None:
                image = draw_target_prediction(image, tracklet.prediction.box)
            image = draw_target_box(image, tracklet.last_detection.box, tracklet.id)
            if draw_skeletons and hasattr(tracklet.last_detection, 'keypoints'):
                image = draw_target_skeleton(image, tracklet.last_detection.keypoints, tracklet.id)
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


def draw_target_box(image, box, id):
    """
    Draw the box with an ID tag for a tracked target.
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param box: A list or numpy array of 4 float numbers. The box of a tracked target in (x1, y1, x2, y2).
    :param id: An integer. The id of the tracked target.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with a new target box drawn.
    """
    image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          _colors[int(id) % _colors.__len__()], thickness=3)
    id_string = '{:d}'.format(int(id))
    id_size, baseline = cv2.getTextSize(id_string, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    image = cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[0] + id_size[0] + 4), int(box[1] + id_size[1]) + 10),
                          _colors[int(id) % _colors.__len__()], thickness=-1)
    image = cv2.putText(image, id_string, (int(box[0] + 2), int(box[1]) + id_size[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=2)
    image = cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), radius=10,
                       color=(0, 0, 255), thickness=-1)
    return image


def draw_target_prediction(image, box):
    """
    Draw the prediction box with an ID tag for a tracked target.
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param box: A list or numpy array of 4 float numbers. The box of a tracked target in (x1, y1, x2, y2).
    :return: A 3D numpy array with shape (h, w, 3). The video frame with a new target prediction box drawn.
    """
    image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (255, 255, 255), thickness=1)
    return image


def draw_target_skeleton(image, keypoints, id):
    """
    Draw the skeleton for a tracked target..
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param keypoints: A 2D list or 2D numpy array of 17 pairs of float numbers. The (x, y) of all body keypoints.
    :param id: An integer. The id of the tracked target.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with its frame number drawn.
    """
    for connection in _keypoint_connections:
        image = cv2.line(image,
                         (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1])),
                         (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1])),
                         color=_colors[int(id) % _colors.__len__()], thickness=2)
    for keypoint in keypoints:
        image = cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 2, (255, 255, 255), thickness=-1)
    return image


def visualize_snapshot(frame, tracker, confirmed_only=True, detected_only=True, draw_predictions=False,
                       draw_skeletons=True):
    """
    Visualize a frame with boxes (and skeletons) of all tracked targets.
    :param frame: A 3D numpy array with shape (h, w, 3). The video frame.
    :param tracker: A Tracker object, of which the active tracklets are to be visualized.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with all targets and frame number drawn.
    """
    image = draw_targets(frame, tracker.tracklets_active,
                         confirmed_only=confirmed_only, detected_only=detected_only, draw_predictions=draw_predictions,
                         draw_skeletons=draw_skeletons)
    image = draw_frame_num(image, tracker.frame_num)
    return image

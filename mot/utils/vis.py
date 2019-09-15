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


def draw_tracklets(image, tracklets, confirmed_only=True):
    """
    Draw the boxes of tracklets.
    :param image: A 3D numpy array with shape (h, w, 3). The video frame.
    :param tracklets: A list of Tracklet objects. The currently active tracklets.
    :return: A 3D numpy array with shape (h, w, 3). The video frame with boxes of tracklets drawn.
    """
    for tracklet in tracklets:
        if (not confirmed_only) or (confirmed_only and tracklet.is_confirmed()):
            image = draw_object(image, tracklet.last_box, tracklet.id)
    return image


def draw_object(image, box, id):
    image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          _colors[int(id) % _colors.__len__()], thickness=3)
    image = cv2.putText(image, '{:d}'.format(int(id)), (int(box[0]), int(box[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, _colors[int(id) % _colors.__len__()], thickness=2)
    image = cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), radius=10,
                       color=(0, 0, 255), thickness=-1)
    return image

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
            box = tracklet.last_box
            image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                  _colors[tracklet.id % _colors.__len__()],
                                  thickness=int(5 * tracklet.ttl / tracklet.max_ttl))
            image = cv2.putText(image, '{:d}'.format(tracklet.id), (int(box[0]), int(box[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, _colors[tracklet.id % _colors.__len__()], thickness=2)
            image = cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), radius=10,
                               color=(0, 0, 255), thickness=-1)
    return image


def generate_video_from_results_file(video_path, results_path, output_path, show=False):
    results = np.loadtxt(results_path)
    capture = cv2.VideoCapture(video_path)
    writer = None
    n_frames = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if writer is None:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MPEG'), 30,
                                     (frame.shape[1], frame.shape[0]))
        for box in results[np.where(results[:, 0] == n_frames)]:
            frame = cv2.rectangle(frame, (int(box[2]), int(box[3])), (int(box[2] + box[4]), int(box[3] + box[5])),
                                  _colors[box[1] % _colors.__len__()], thickness=3)
            frame = cv2.putText(frame, '{:d}'.format(box[1]), (int(box[0]), int(box[1]) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, _colors[box[0] % _colors.__len__()], thickness=2)
            frame = cv2.circle(frame, (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)), radius=10,
                               color=(0, 0, 255), thickness=-1)
        writer.write(frame)
        if show:
            cv2.imshow('Output', frame)
            cv2.waitKey(1)
    if writer is not None:
        writer.close()

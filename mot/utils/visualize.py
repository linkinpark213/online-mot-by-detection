import cv2
import datetime
import numpy as np
from typing import List, Union

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

__all__ = ['snapshot_from_tracker', 'snapshot_from_results', 'snapshot_from_detection']


def _draw_detections(image: np.ndarray, detections: List):
    """
    Draw detection bounding boxes.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        detections: A list of Detection objects. All detected tracklets.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with boxes of detection boxes drawn.
    """
    for detection in detections:
        box = detection.box
        image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness=1)
    return image


def _draw_targets(image: np.ndarray, tracklets: List, confirmed_only: bool = True, detected_only: bool = True,
                  draw_centers: bool = False, draw_predictions: bool = False, draw_trajectory: bool = True,
                  draw_masks: bool = True, draw_skeletons: bool = True) -> np.ndarray:
    """
    Draw the boxes of targets.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        tracklets: A list of Tracklet objects. The currently active tracklets.
        confirmed_only: A boolean value. Set to True to only visualize targets that are confirmed.
        detected_only: A boolean value. Set to True to only disable visualizing targets that are only predicted.
        draw_centers: A boolean value. Set to True to visualize center points of targets.
        draw_predictions: A boolean value. Set to True to visualize predictions of targets too, if it's available.
        draw_trajectory: A boolean value. Set to True to visualize trajectories o targets.
        draw_masks: A boolean value. Set to True to visualize target masks, if it's available.
        draw_skeletons: A boolean value. Set to True to visualize target body keypoints, if it's available.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with boxes of tracked targets drawn.
    """
    for tracklet in tracklets:
        if (tracklet.is_confirmed() or not confirmed_only) and (tracklet.is_detected() or not detected_only):
            if draw_predictions and tracklet.prediction is not None:
                image = _draw_target_prediction(image, tracklet.prediction.box)
            if draw_trajectory:
                image = _draw_target_trajectory(image, tracklet)
            if draw_masks:
                image = _draw_target_mask(image, tracklet.last_detection.mask, tracklet.id)
            if draw_skeletons and hasattr(tracklet.last_detection, 'keypoints'):
                image = _draw_target_skeleton(image, tracklet.last_detection.keypoints, tracklet.id)
            # Box over everything!
            image = _draw_target_box(image, tracklet.last_detection.box, tracklet.id, globalID=tracklet.globalID,
                                     draw_center=draw_centers)
    return image


def _draw_frame_num(image: np.ndarray, frame_num: int, inverse_color: bool = True) -> np.ndarray:
    """
    Draw the frame number at the top-left corner of the frame.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        frame_num: Frame number.
        inverse_color: A boolean value. If True, use black/white colors for better visibility; else use yellow color.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with its frame number drawn.
    """
    if inverse_color:
        blank = np.zeros(image.shape[:2], dtype=np.uint8)
        blank = cv2.putText(blank, '{}'.format(frame_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)
        inds = np.where(blank != 0)
        colors = np.sum(image[inds], axis=1)
        colors = (0.5 + (255 - colors) / 255.).astype(np.int) * 255
        image[inds] = np.stack([colors, colors, colors], axis=1)
    else:
        cv2.putText(image, '{}'.format(frame_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)
    return image


def _draw_current_time(image: np.ndarray, inverse_color: bool = True) -> np.ndarray:
    """
    Draw the local time at the top-left corner of the frame.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        inverse_color: A boolean value. If True, use black/white colors for better visibility; else use yellow color.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with the current local time drawn.
    """
    current_time = str(datetime.datetime.now())
    if inverse_color:
        blank = np.zeros(image.shape[:2], dtype=np.uint8)
        blank = cv2.putText(blank, current_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, thickness=2)
        inds = np.where(blank != 0)
        colors = np.sum(image[inds], axis=1)
        colors = (0.5 + (255 - colors) / 255.).astype(np.int) * 255
        image[inds] = np.stack([colors, colors, colors], axis=1)
    else:
        cv2.putText(image, current_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), thickness=1)
    return image


def _draw_target_box(image: np.ndarray, box: Union[List[float], np.ndarray], id: int, globalID: int = -1,
                     draw_center: bool = False, ) -> np.ndarray:
    """
    Draw the box with an ID tag for a tracked target.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        box: A list or numpy array of 4 float numbers. The box of a tracked target in (x1, y1, x2, y2).
        id: An integer. The id of the tracked target.
        globalID: An integer. The global id of the tracked target. -1 means the global id is unavailable.
        draw_center: Set to true to draw a red dot at the target center.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with a new target box drawn.
    """
    image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          _colors[int(id) % _colors.__len__()], thickness=3)
    id_string = '{:d}'.format(int(id))
    id_size, baseline = cv2.getTextSize(id_string, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    image = cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[0] + id_size[0] + 4), int(box[1] + id_size[1]) + 10),
                          _colors[int(id) % _colors.__len__()], thickness=-1)
    image = cv2.putText(image, id_string, (int(box[0] + 2), int(box[1] + id_size[1] + 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=2)
    if globalID >= 0:
        id_string = '{:d}'.format(globalID)
        id_size, baseline = cv2.getTextSize(id_string, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        image = cv2.rectangle(image, (int(box[2] - id_size[0] - 4), int(box[3] - id_size[1] - 10)),
                              (int(box[2]), int(box[3])),
                              _colors[int(id) % _colors.__len__()], thickness=-1)
        image = cv2.putText(image, id_string, (int(box[2] - id_size[0] - 2), int(box[3] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=2)

    if draw_center:
        image = cv2.circle(image, (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)), radius=10,
                           color=(0, 0, 255), thickness=-1)
    return image


def _draw_target_prediction(image: np.ndarray, box: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Draw the prediction box with an ID tag for a tracked target.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        box: A list or numpy array of 4 float numbers. The box of a tracked target in (x1, y1, x2, y2).

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with a new target prediction box drawn.
    """
    image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          (255, 255, 255), thickness=1)
    return image


def _draw_target_trajectory(image: np.ndarray, tracklet, mix_exponential: float = 0.95,
                            block_radius: int = 4, max_steps: int = 300) -> np.ndarray:
    """
    Draw the trajectory of a tracked target according to its old detections.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        tracklet: A Tracklet object with the arrtibute 'detection_history'.
        mix_exponential: A floating number. The percentage that the color of trajectory points fade as time goes by.
        block_radius: An integer. Size of the target trajectory points will be block_size * 2 + 1.
        max_steps: An integer. The max length of any trajectory visualized.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with the trajectory drawn.
    """
    mix_factor = 1
    br = block_radius
    for i, (frame_id, detection) in enumerate(reversed(tracklet.detection_history)):
        box = [int(_) for _ in detection.box]
        y, x = box[3], (box[0] + box[2]) // 2
        color = _colors[int(tracklet.id) % _colors.__len__()]

        for c in range(3):
            image[y - br: y + br + 1, x - br: x + br + 1, c] = image[y - br: y + br + 1, x - br:x + br + 1, c] * (
                    1 - mix_factor) + color[c] * mix_factor
        mix_factor = mix_factor * mix_exponential

        if i > max_steps:
            break

    return image


def _draw_target_mask(image: np.ndarray, mask: np.ndarray, id: int) -> np.ndarray:
    """
    Draw the skeleton for a tracked target.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        mask: A 2D numpy array with type bool and same spatial shape as the image but no color channel.
        id: An integer. The id of the tracked target.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with its frame number drawn.
    """
    if mask is not None:
        image[mask] = image[mask] // 2 + np.array(_colors[id % len(_colors)]) // 2
    return image


def _draw_target_skeleton(image: np.ndarray, keypoints: Union[List[List[float]], np.ndarray], id: int) -> np.ndarray:
    """
    Draw the skeleton for a tracked target.

    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        keypoints: A 2D list or 2D numpy array of 17 pairs of float numbers. The (x, y) of all body keypoints.
        id: An integer. The id of the tracked target.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with its frame number drawn.
    """
    for connection in _keypoint_connections:
        image = cv2.line(image,
                         (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1])),
                         (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1])),
                         color=_colors[int(id) % _colors.__len__()], thickness=2)
    for keypoint in keypoints:
        image = cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 2, (255, 255, 255), thickness=-1)
    return image


def _draw_association(image: np.ndarray, tracklets: List) -> np.ndarray:
    """
    Draw tracklets at the bottom and link them to their current bounding boxes.
    Args:
        image: A 3D numpy array with shape (h, w, 3). The video frame.
        tracklets: A list of Tracklet objects. The tracklets to visualize.

    Returns:
        A 3D numpy array with shape (h + 2 * w // 40, w, 3). The video frame with target patches drawn.
    """
    img_h, img_w, _ = image.shape
    grid_w = img_w // max(40, len(tracklets))
    grid_h = (img_w // 40) * 2
    id_size, baseline = cv2.getTextSize('0', cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    image = np.concatenate((image, np.zeros((grid_h + id_size[1], img_w, 3)).astype(np.uint8)), axis=0)
    for i, tracklet in enumerate(tracklets):
        # First draw tracklet in the bottom
        assert 'patch' in tracklet.feature.keys(), 'An ImagePatchEncoder with name `patch` should be enabled to visualize tracklets'
        patch = tracklet.feature_history[-1][1]['patch']
        patch_h, patch_w, _ = patch.shape
        if patch_h > patch_w * 2:
            patch = cv2.resize(patch, (int(patch_w * (grid_h / patch_h)), grid_h))
        else:
            patch = cv2.resize(patch, (grid_w, int(patch_h * (grid_w / patch_w))))
        grid_t, grid_b = img_h, img_h + patch.shape[0]
        grid_l, grid_r = grid_w * i + (grid_w - patch.shape[1]) // 2, grid_w * i + (grid_w - patch.shape[1]) // 2 + \
                         patch.shape[1]

        image[grid_t: grid_b, grid_l:grid_r, :] = patch
        image = cv2.rectangle(image, (grid_l, grid_t), (grid_r - 2, grid_b - 2),
                              _colors[int(tracklet.id) % len(_colors)],
                              thickness=2)

        # Then connect tracklet with the new detection
        if tracklet.is_detected() and tracklet.is_confirmed():
            image = cv2.line(image, ((grid_l + grid_r) // 2, img_h),
                             (int((tracklet.last_detection.box[0] + tracklet.last_detection.box[2]) // 2),
                              int(tracklet.last_detection.box[3])),
                             _colors[int(tracklet.id) % len(_colors)], thickness=2)
    return image


def snapshot_from_tracker(tracker, confirmed_only: bool = True, detected_only: bool = True, draw_centers: bool = False,
                          draw_detections: bool = False, draw_predictions: bool = False, draw_masks: bool = False,
                          draw_skeletons: bool = False, draw_association: bool = False, draw_frame_num: bool = True,
                          draw_current_time: bool = False, **kwargs) -> np.ndarray:
    """
    Visualize a frame with boxes (and skeletons) of all tracked targets.

    Args:
        tracker: A Tracker object, of which the active tracklets are to be visualized.
        confirmed_only: A boolean value. Set to True to only visualize targets that are confirmed.
        detected_only: A boolean value. Set to True to only disable visualizing targets that are only predicted.
        draw_centers: A boolean value. Set to True to visualize center points of targets.
        draw_detections: A boolean value. Set to True to visualize detection boxes.
        draw_predictions: A boolean value. Set to True to visualize predictions of targets too, if it's available.
        draw_masks: A boolean value. Set to True to visualize target masks, if it's available.
        draw_skeletons: A boolean value. Set to True to visualize target body keypoints, if it's available.
        draw_association: A boolean value. Set to True to visualize active tracklets and their connections with detections in the frame.
        draw_frame_num: A boolean value. Set to True to visualize current frame number.
        draw_current_time: A boolean value. Set to True to visualize current time.

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with all targets and frame number drawn.
    """
    image = tracker.frame
    # Draw detection boxes first so matched boxes will be covered by target box.
    if draw_detections:
        image = _draw_detections(image, tracker.latest_detections)

    image = _draw_targets(image, tracker.tracklets_active,
                          confirmed_only=confirmed_only, detected_only=detected_only, draw_centers=draw_centers,
                          draw_predictions=draw_predictions, draw_masks=draw_masks, draw_skeletons=draw_skeletons)
    if draw_frame_num:
        image = _draw_frame_num(image, tracker.frame_num)
    if draw_current_time:
        image = _draw_current_time(image)
    if draw_association:
        image = _draw_association(image, tracker.tracklets_active)
    return image


def snapshot_from_results(frame: np.ndarray, boxes: np.ndarray, frame_num: int = -1) -> np.ndarray:
    """
    Visualize a frame with a line of tracker output.

    Args:
        frame: A 3D numpy array with shape (h, w, 3). The video frame.
        boxes: A 2D numpy array with shape (b, 6+). The boxes in the tracker output file.
            6 = 1 (Frame number) + 1 (ID) + 4 (l, t, w, h)
        frame_num: An integer. The current frame number. (Default -1 means unavailable)

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with all targets drawn.
    """
    xyxy = boxes.copy()
    xyxy[:, 4] = xyxy[:, 2] + xyxy[:, 4]
    xyxy[:, 5] = xyxy[:, 3] + xyxy[:, 5]
    for box in xyxy:
        frame = _draw_target_box(frame, box[2:6], box[1])
    if frame_num >= 0:
        frame = _draw_frame_num(frame, frame_num)
    return frame


def snapshot_from_detection(frame: np.ndarray, boxes: np.ndarray, frame_num: int = -1) -> np.ndarray:
    """
    Visualize a frame with detection boxes.

    Args:
        frame: A 3D numpy array with shape (h, w, 3). The video frame.
        boxes: A 2D numpy array with shape (b, 6+). The boxes in the tracker output file.
            6 = 1 (Frame number) + 1 (ID) + 4 (x1, y1, x2, y2)
        frame_num: An integer. The current frame number. (Default -1 means unavailable)

    Returns:
        A 3D numpy array with shape (h, w, 3). The video frame with all targets and frame number drawn.
    """
    for box in boxes:
        frame = _draw_target_box(frame, box[2:6], box[1])
    if frame_num >= 0:
        frame = _draw_frame_num(frame, frame_num)
    return frame

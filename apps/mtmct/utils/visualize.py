import cv2
import numpy as np
from typing import Tuple, Dict

from mot.utils.visualize import _colors


def snapshot_from_pool(identity_pool: Dict[int, object], resolution: Tuple[int, int], frame_num: int) -> np.ndarray:
    img_w, img_h = resolution
    image = np.zeros((img_h, img_w, 3)).astype(np.uint8)
    globalIDs = list(identity_pool.keys())
    N = len(globalIDs)
    if N > 0:
        if N > 1:
            size = int((np.sqrt((img_w + img_h) ** 2 + 4 * (N - 1) * img_w * img_h) - img_w - img_h) / (2 * N - 2))
            size = min(size, 256)
        else:
            size = 256
        n_in_row, n_in_col = img_w // size, img_h // size

        for i in range(N):
            row, col = i // n_in_row, i % n_in_row
            identity = identity_pool[globalIDs[i]]
            patch = cv2.resize(identity.images[list(identity.images.keys())[0]], (size, size))
            image[size * row: size * row + size, size * col: size * col + size, :] = patch
            cv2.putText(image, str(identity.globalID), (size * col + 10, size * row + 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                        color=_colors[identity.globalID % len(_colors)],
                        thickness=3)

    image = cv2.putText(image, str(frame_num), (img_w - 300, img_h - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, color=(0, 255, 255), thickness=3)

    return image

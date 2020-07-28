import numpy as np
from pycocotools import mask as cocomask


def snapshot_to_mot(tracker, confirmed_only=False, detected_only=True, mots=False):
    data = ''
    for tracklet in tracker.tracklets_active:
        if (tracklet.is_confirmed() or not confirmed_only) and (tracklet.is_detected() or not detected_only):
            if not mots:
                box = tracklet.last_detection.box
                score = tracklet.last_detection.score
                data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                                  tracklet.id,
                                                                                                  box[0],
                                                                                                  box[1],
                                                                                                  box[2] - box[0],
                                                                                                  box[3] - box[1],
                                                                                                  score)
            else:
                box = tracklet.last_detection.box
                score = tracklet.last_detection.score
                mask = tracklet.last_detection.mask
                assert type(mask) is np.ndarray, 'Expected an numpy array for mask, but got {}'.format(type(mask))
                h, w = mask.shape
                mask = np.asfortranarray(mask).astype(np.uint8)
                data += '{:d} {:d} 2 {:d} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:s}\n'.format(tracker.frame_num,
                                                                                                 tracklet.id,
                                                                                                 h,
                                                                                                 w,
                                                                                                 box[0],
                                                                                                 box[1],
                                                                                                 box[2] - box[0],
                                                                                                 box[3] - box[1],
                                                                                                 score,
                                                                                                 str(cocomask.encode(
                                                                                                     mask)[
                                                                                                     'counts'].decode(
                                                                                                     'utf-8')))
    return data

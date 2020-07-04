import os
import cv2
import logging
import argparse
import numpy as np

import mot.utils
from mot.utils import get_capture
from mot.tracker import build_tracker

from apps.mtmct.tracker import MCTracker
from apps.mtmct.utils.visualize import snapshot_from_pool


def run_demo(mtracker, args, **kwargs):
    result_writers = {}
    if args.save_video != '' and not os.path.isdir(os.path.abspath(os.path.dirname(args.save_video))):
        os.makedirs(os.path.abspath(os.path.dirname(args.save_video)))
    if args.save_result != '' and not os.path.isdir(args.save_result):
        os.makedirs(args.save_result)

    mc_video_writer = mot.utils.get_video_writer(args.save_video, 1920, 1080, fps=args.fps)
    mc_result_writer = open(os.path.join(args.save_result, 'out_mc.txt'), 'w+')
    for capID, capture in captures.items():
        result_writers[capID] = mot.utils.get_result_writer(os.path.join(args.save_result, 'out_{}.txt').format(capID))

    logger = logging.getLogger('MTMCT')
    logging.basicConfig(level=logging.INFO)
    if args.save_log != '':
        logger.addHandler(logging.FileHandler(args.save_log, mode='w+'))

    while True:
        ret = mtracker.tick()

        if not ret:
            break

        msg = 'Frame #{}: {} ID(s) in pool'.format(mtracker.frame_num, len(mtracker.identity_pool))
        for camID, state in mtracker.stracker_states.items():
            msg += '| cam #{} - {} target(s)'.format(camID, len(state.tracklets_active))
        logger.info(msg)

        # Save results for each frame, both single-camera and multi-camera.
        cleared_identities = mtracker.clear_inactive_identities()
        for globalID, identity in cleared_identities:
            for (camID, localID), tracklet in identity.tracklet_dict.items():
                mc_result_writer.write('{} {} {}\n'.format(globalID, camID, localID))

        views = []
        for i, stracker_state in mtracker.stracker_states.items():
            snapshot = mot.utils.snapshot_from_tracker(stracker_state, draw_frame_num=False)
            snapshot = cv2.putText(snapshot, 'CAMERA {}'.format(i), (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), thickness=3)
            views.append(snapshot)
            result_writers[i].write(mot.utils.snapshot_to_mot(stracker_state))

        pool_view = snapshot_from_pool(mtracker.identity_pool, (1920, 1080), frame_num=mtracker.frame_num)
        full_view = np.vstack((np.hstack((views[0], views[1])), np.hstack((views[2], pool_view))))
        full_view = cv2.resize(full_view, (1920, 1080))
        mc_video_writer.write(full_view)

    # Save results when terminating.
    mtracker.terminate()
    for globalID, identity in mtracker.identity_pool.items():
        for (camID, localID), tracklet in identity.tracklet_dict.items():
            mc_result_writer.write('{} {} {}\n'.format(globalID, camID, localID))

    for camID, _ in result_writers.items():
        result_writers[camID].close()
    mc_video_writer.release()
    mc_result_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('--save-result', default='', required=False,
                        help='Path to the output tracking result directory. Leave it blank to disable.')
    parser.add_argument('--save-log', default='', required=False,
                        help='Path to the output log file. Leave it blank to disable.')
    parser.add_argument('--fps', default=60, type=int, required=False, help='Frames per second of the output video')
    args = parser.parse_args()

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    captures = {
        1: get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/images_10fps/camera1'),
        2: get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/images_10fps/camera2'),
        8: get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/images_10fps/camera8'),
    }

    mtracker = MCTracker(cfg.tracker,
                         captures,
                         cluster_freq=60,
                         max_ttl=600,
                         max_local_overlap=10,
                         max_reid_distance=0.4,
                         n_feature_samples=8)

    run_demo(mtracker, args)

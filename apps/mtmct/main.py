import os
import argparse

import mot.utils
from mot.utils import get_capture
from mot.tracker import build_tracker

from apps.mtmct.tracker import MCTracker


def run_demo(mtracker, args, **kwargs):
    video_writers = {}
    result_writers = {}
    if args.save_video != '' and not os.path.isdir(args.save_video):
        os.makedirs(args.save_video)
    if args.save_result != '' and not os.path.isdir(args.save_result):
        os.makedirs(args.save_result)
    mc_result_writer = open(os.path.join(args.save_result, 'out_mc.txt'), 'w+')
    for capID, capture in captures.items():
        video_writers[capID] = mot.utils.get_video_writer(os.path.join(args.save_video, 'out_{}.mp4'.format(capID)),
                                                          1920, 1080, fps=60)
        result_writers[capID] = mot.utils.get_result_writer(os.path.join(args.save_result, 'out_{}.txt').format(capID))

    while True:
        ret = mtracker.tick()

        if not ret:
            break

        print('MTMCTracker: frame #{}, {} ID(s) in pool'.format(mtracker.frame_num, len(mtracker.identity_pool)),
              end=', ')
        for camID, state in mtracker.stracker_states.items():
            print('| cam #{} - {} target(s)'.format(camID, len(state.tracklets_active)), end=' ')
        print()

        # Save results for each frame, both single-camera and multi-camera.
        cleared_identities = mtracker.clear_old_identities()
        for globalID, identity in cleared_identities:
            for (camID, localID), tracklet in identity.tracklets.items():
                mc_result_writer.write('{} {} {}\n'.format(globalID, camID, localID))

        for i, stracker_state in mtracker.stracker_states.items():
            video_writers[i].write(mot.utils.snapshot_from_tracker(stracker_state))

            result_writers[i].write(mot.utils.snapshot_to_mot(stracker_state))

    # Save results when terminating.
    mtracker.terminate()
    for globalID, identity in mtracker.identity_pool.items():
        for (camID, localID), tracklet in identity.tracklets.items():
            mc_result_writer.write('{} {} {}\n'.format(globalID, camID, localID))

    for camID, _ in video_writers.items():
        video_writers[camID].release()
        result_writers[camID].close()
    mc_result_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video directory. Leave it blank to disable.')
    parser.add_argument('--save-result', default='', required=False,
                        help='Path to the output tracking result directory. Leave it blank to disable.')
    args = parser.parse_args()

    cfg = mot.utils.cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    captures = {
        1: get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/images_tiny/camera1'),
        2: get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/images_tiny/camera2'),
        8: get_capture('/mnt/nasbi/no-backups/datasets/object_tracking/DukeMTMC/images_tiny/camera8'),
    }

    mtracker = MCTracker(cfg.tracker,
                         captures,
                         cluster_freq=60,
                         max_ttl=6000,
                         max_local_overlap=10,
                         max_reid_distance=0.25,
                         n_feature_samples=8)

    run_demo(mtracker, args)

import os
import cv2
import logging
import argparse

from mot.detect import MOTPublicDetector
from mot.tracker import build_tracker, Tracker
from mot.utils import get_capture, get_video_writer, get_result_writer, snapshot_from_tracker, cfg_from_file, Config


def evaluate_mot_online(tracker: Tracker, mot_subset_path: str, output_path: str = 'results',
                        output_video_path: str = 'videos', **kwargs):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_video_path):
        os.mkdir(output_video_path)

    for sequence in sorted(os.listdir(mot_subset_path)):
        print('Processing sequence {}'.format(sequence))
        capture = get_capture(os.path.join(mot_subset_path, sequence, 'img1'))
        video_writer = get_video_writer(os.path.join(output_video_path, sequence + '.mp4'),
                                        capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                        capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result_writer = get_result_writer(os.path.join(output_path, sequence + '.txt'))

        # Replace detector with MOT public detection
        del tracker.detector
        tracker.detector = MOTPublicDetector(
            det_file_path=os.path.join(mot_subset_path, sequence, 'det', 'det.txt'),
            conf_threshold=args.det_conf_thresh,
        )

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            tracker.tick(frame)
            image = snapshot_from_tracker(frame, tracker, **kwargs)

            # Write to video if demanded.
            video_writer.write(image)

            # Write to result file if demanded.
            result_writer.write(snapshot_to_mot(tracker))

        tracker.terminate()
        tracker.clear()

        # Close writers after tracking
        video_writer.release()
        result_writer.close()
        print('Results saved to {}/{}.txt'.format(output_path, sequence))


def snapshot_to_mot(tracker, time_lived_threshold=1, detected_only=True):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and (tracklet.is_detected() or not detected_only):
            box = tracklet.last_detection.box
            score = tracklet.last_detection.score
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                              tracklet.id,
                                                                                              box[0],
                                                                                              box[1],
                                                                                              box[2] - box[0],
                                                                                              box[3] - box[1],
                                                                                              score)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--mot-subset-path', required=True,
                        help='Path to the test video file or directory of test images.')
    parser.add_argument('--det-conf-thresh', required=False, type=float, default=0.5,
                        help='Confidence threshold of detections')
    parser.add_argument('--output-path', required=True,
                        help='Path to the output tracking result file.')
    parser.add_argument('--save-video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable saving video.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('MOT')
    logger.setLevel(logging.INFO)

    cfg = cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict(ignore_keywords=True)

    tracker = build_tracker(cfg.tracker)

    evaluate_mot_online(tracker, args.mot_subset_path, output_path=args.output_path, output_video_path=args.save_video,
                        **kwargs)

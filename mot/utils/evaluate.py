import os
import cv2
import logging
import numpy as np
import mot.detect
import mot.metric
import mot.associate
import mot.utils.io
import mot.utils.offline
import mot.utils.visualize


def evaluate_mot_online(tracker, mot_subset_path, output_path='results',
                        output_video_path='videos'):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_video_path):
        os.mkdir(output_video_path)
    for sequence in os.listdir(mot_subset_path):
        print('Processing sequence {}'.format(sequence))
        capture = mot.utils.get_capture(os.path.join(mot_subset_path, sequence, 'img1'))
        video_writer = mot.utils.get_video_writer(os.path.join(output_video_path, sequence + '.mp4'),
                                                  capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                  capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result_writer = mot.utils.get_result_writer(os.path.join(output_path, sequence + '.txt'))
        detector = mot.detect.MOTPublicDetector(os.path.join(mot_subset_path, sequence, 'det', 'det.txt'))
        tracker.detector = detector

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            tracker.tick(frame)
            image = mot.utils.visualize_snapshot(frame, tracker, draw_predictions=True, draw_skeletons=False)

            # Write to video if demanded.
            video_writer.write(image)

            # Write to result file if demanded.
            result_writer.write(mot.utils.snapshot_to_mot(tracker))

        tracker.terminate()
        tracker.clear()

        # Close writers after tracking
        video_writer.release()
        result_writer.close()
        print('Results saved to {}/{}.txt'.format(output_path, sequence))


def snapshot_to_mot(tracker, time_lived_threshold=1, ttl_threshold=3, detected_only=True):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and (
                tracklet.is_detected() or not detected_only):
            box = tracklet.last_detection.box
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                          tracklet.id,
                                                                                          box[0],
                                                                                          box[1],
                                                                                          box[2] - box[0],
                                                                                          box[3] - box[1])
    return data

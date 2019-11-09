import os
import cv2
import logging
import numpy as np
import mot.detect
import mot.metric
import mot.associate
import mot.utils.offline
import mot.utils.visualize


def evaluate_mot_online(tracker, mot_subset_path, output_path='results',
                        output_video_path='videos', show_result=False):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_video_path):
        os.mkdir(output_video_path)
    for sequence in os.listdir(mot_subset_path):
        print('Processing sequence {}'.format(sequence))
        result_file = open(os.path.join(output_path, sequence + '.txt'), 'w+')
        video_writer = None
        detector = mot.detect.MOTPublicDetector(os.path.join(mot_subset_path, sequence, 'det', 'det.txt'))
        tracker.clear()
        tracker.detector = detector

        frame_filenames = os.listdir(os.path.join(mot_subset_path, sequence, 'img1'))
        frame_filenames.sort()
        for i in range(frame_filenames.__len__()):
            if video_writer is None:
                video_writer = cv2.VideoWriter(os.path.join(output_video_path, '{}.mp4'.format(sequence)),
                                               cv2.VideoWriter_fourcc(*'mp4v'),
                                               30, (image.shape[1], image.shape[0]))
            image = cv2.imread(os.path.join(mot_subset_path, sequence, 'img1', frame_filenames[i]))
            result_file.write(snapshot_to_mot(tracker))
            tracker.tick(image)
            image = mot.utils.visualize.draw_tracklets(image, tracker.tracklets_active)

            video_writer.write(image)
            if show_result:
                image = cv2.resize(image, (960, 540))
                cv2.imshow(sequence, image)
                key = cv2.waitKey(1)
                if key == 27:
                    return

        cv2.destroyAllWindows()
        video_writer.release()
        result_file.close()
        print('Results saved to {}/{}.txt'.format(output_path, sequence))


def snapshot_to_mot(tracker, time_lived_threshold=1, ttl_threshold=3, detected_only=True):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and (
                tracklet.is_detected() or not detected_only):
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                          tracklet.id,
                                                                                          tracklet.last_detection.box[0],
                                                                                          tracklet.last_detection.box[1],
                                                                                          tracklet.last_detection.box[2],
                                                                                          tracklet.last_detection.box[3])
    return data

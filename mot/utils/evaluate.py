import os
import cv2
import logging
import numpy as np
import mot.utils.vis
import mot.detect
import mot.metric
import mot.associate
import mot.utils.offline


def evaluate_zhejiang(tracker, videos_path, detections_path, output_path='results',
                      output_video_path='videos', level=1, online=True, show_result=False):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(output_video_path):
        os.mkdir(output_video_path)
    if level == 1:
        sequences = ['a1', 'a2', 'a3', 'a4', 'a5']
    elif level == 2:
        sequences = ['b1', 'b2', 'b3', 'b4', 'b5']
    else:
        raise AssertionError('Level should be 1 or 2')
    for sequence in sequences:
        logging.info('Processing sequence {}'.format(sequence))
        result_file = open(os.path.join(output_path, sequence + '.txt'), 'w+')
        capture = cv2.VideoCapture(os.path.join(videos_path, sequence + '.mp4'))
        video_writer = None
        detector = mot.detect.ZhejiangFakeDetector(os.path.join(detections_path, sequence + '.txt'),
                                                   conf_threshold=0.5,
                                                   height_threshold=50,
                                                   hw_ratio_lower_bound=1,
                                                   hw_ratio_upper_bound=5)
        tracker.clear()
        tracker.detector = detector

        while True:
            ret, image = capture.read()
            if not ret:
                break
            if video_writer is None:
                video_writer = cv2.VideoWriter(os.path.join(output_video_path, '{}.mp4'.format(sequence)),
                                               cv2.VideoWriter_fourcc(*'mp4v'),
                                               30, (image.shape[1], image.shape[0]))
            if online:
                result_file.write(snapshot_to_zhejiang(tracker))
            tracker.tick(image)
            image = mot.utils.vis.draw_tracklets(image, tracker.tracklets_active)

            video_writer.write(image)
            if show_result:
                image = cv2.resize(image, (960, 540))
                cv2.imshow(sequence, image)
                key = cv2.waitKey(1)
                if key == 27:
                    return

        if not online:
            for tracklet in tracker.tracklets_active:
                tracker.tracklets_finished.append(tracklet)
            all_trajectories = []
            for tracklet in tracker.tracklets_finished:
                all_trajectories.append((tracklet.id, tracklet.box_history))
            # all_trajectories = mot.utils.offline.fill_gaps(all_trajectories, max_gap=10)
            # all_trajectories = mot.utils.offline.remove_short_tracks(all_trajectories, min_time_lived=30)
            result_file.write(trajectories_to_zhejiang(all_trajectories))

        cv2.destroyAllWindows()
        video_writer.release()
        result_file.close()
        logging.info('Results saved to {}/{}.txt'.format(output_path, sequence))


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
            image = mot.utils.vis.draw_tracklets(image, tracker.tracklets_active)

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


def trajectories_to_zhejiang(trajectories):
    data = ''
    trajectories = [trajectories[i] for i in np.argsort(np.array([id for id, trajectory in trajectories]))]
    for id, trajectory in trajectories:
        for frame, box in trajectory:
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(frame, id, box[0], box[1], box[2] - box[0],
                                                                          box[3] - box[1])
    return data


def snapshot_to_zhejiang(tracker, time_lived_threshold=1, ttl_threshold=3):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and tracklet.detected:
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(tracker.frame_num,
                                                                          tracklet.id,
                                                                          tracklet.last_box[0],
                                                                          tracklet.last_box[1],
                                                                          tracklet.last_box[2] - tracklet.last_box[0],
                                                                          tracklet.last_box[3] - tracklet.last_box[1])
    return data


def snapshot_to_mot(tracker, time_lived_threshold=1, ttl_threshold=3):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and tracklet.detected:
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                          tracklet.id,
                                                                                          tracklet.last_box[0],
                                                                                          tracklet.last_box[1],
                                                                                          tracklet.last_box[2],
                                                                                          tracklet.last_box[3])
    return data

import os
import cv2
import utils.vis
import mot.detect
import mot.metric
import mot.associate


def evaluate_zhejiang_online(tracker, videos_path, detections_path, output_path='results', show_result=False):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for sequence in ['a1', 'a2', 'a3', 'a4', 'a5']:
        print('Processing sequence {}'.format(sequence))
        result_file = open(os.path.join(output_path, sequence + '.txt'), 'w+')
        capture = cv2.VideoCapture(os.path.join(videos_path, sequence + '.mp4'))
        detector = mot.detect.ZhejiangFakeDetector(os.path.join(detections_path, sequence + '.txt'))
        tracker.clear()
        tracker.detector = detector

        while True:
            ret, image = capture.read()
            if not ret:
                break
            result_file.write(to_zhejiang_evaluate_data(tracker))
            tracker.tick(image)
            image = utils.vis.draw_tracklets(image, tracker.tracklets_active)

            if show_result:
                image = cv2.resize(image, (960, 540))
                cv2.imshow(sequence, image)
                key = cv2.waitKey(1)
                if key == 27:
                    return

        cv2.destroyAllWindows()
        result_file.close()
        print('Results saved to {}/{}.txt'.format(output_path, sequence))


def evaluate_mot_online(tracker, mot_subset_path, output_path='results', show_result=False):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for sequence in os.listdir(mot_subset_path):
        print('Processing sequence {}'.format(sequence))
        result_file = open(os.path.join(output_path, sequence + '.txt'), 'w+')
        detector = mot.detect.MOTPublicDetector(os.path.join(mot_subset_path, sequence, 'det', 'det.txt'))
        tracker.clear()
        tracker.detector = detector

        frame_filenames = os.listdir(os.path.join(mot_subset_path, sequence, 'img1'))
        frame_filenames.sort()
        for i in range(frame_filenames.__len__()):
            image = cv2.imread(os.path.join(mot_subset_path, sequence, 'img1', frame_filenames[i]))
            result_file.write(to_mot_evaluate_data(tracker))
            tracker.tick(image)
            image = utils.vis.draw_tracklets(image, tracker.tracklets_active)

            if show_result:
                image = cv2.resize(image, (960, 540))
                cv2.imshow(sequence, image)
                key = cv2.waitKey(1)
                if key == 27:
                    return

        cv2.destroyAllWindows()
        result_file.close()
        print('Results saved to {}/{}.txt'.format(output_path, sequence))


def to_zhejiang_evaluate_data(tracker, time_lived_threshold=1, ttl_threshold=3):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold:
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(tracker.frame_num,
                                                                          tracklet.id,
                                                                          tracklet.last_box[0],
                                                                          tracklet.last_box[1],
                                                                          tracklet.last_box[2],
                                                                          tracklet.last_box[3])
    return data


def to_mot_evaluate_data(tracker, time_lived_threshold=1, ttl_threshold=3):
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold:
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                          tracklet.id,
                                                                                          tracklet.last_box[0],
                                                                                          tracklet.last_box[1],
                                                                                          tracklet.last_box[2],
                                                                                          tracklet.last_box[3])
    return data

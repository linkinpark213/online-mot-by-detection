import os
import cv2
import utils.vis
import mot.detect
import mot.metric
import mot.associate


def evaluate_mot(tracker, mot_subset_path, output_path=''):
    for sequence in os.listdir(mot_subset_path):
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

            cv2.imshow(sequence, image)
            key = cv2.waitKey(1)
            if key == 27:
                return

        cv2.destroyAllWindows()
        result_file.close()


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

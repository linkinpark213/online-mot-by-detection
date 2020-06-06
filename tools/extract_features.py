import os
import cv2
import random
import argparse
import numpy as np
from typing import List
import scipy.cluster.hierarchy as sch

from mot.utils import cfg_from_file
from mot.encode import build_encoder
from mot.structures import Detection


def sample_detections(tracks: np.ndarray, target_id: int, sample_rate: float, score_threshold: float) -> np.ndarray:
    all_detections = tracks[np.where(tracks[:, 1] == target_id)]
    inds = np.where(all_detections[:, 6] >= score_threshold)[0]
    if not inds.any():
        return [all_detections[np.argmax(all_detections[:, 6])]]

    all_detections = all_detections[inds]
    detections = []
    for detection in all_detections:
        if sample_rate > random.random():
            detections.append(detection)
    if len(detections) == 0:
        detections.append(all_detections[np.argmax(all_detections[:, 6])])
    print('Target #{} has {} detections and {} are sampled'.format(target_id, len(all_detections), len(detections)))
    return np.stack(detections, axis=0)


def filter_features(features: np.ndarray, max_cluster_distance: float) -> List[int]:
    M = np.eye(len(features))
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            M[i][j] = 1 - np.dot(features[i], features[j]) / (
                    (np.linalg.norm(features[i]) * np.linalg.norm(features[j])) + 1e-16)

    # print('Similarity matrix')
    # for line in M:
    #     print(line)

    iu = np.triu_indices(len(features), 1, len(features))
    M = M[iu]
    linkage = sch.linkage(M, method='average', metric='cosine')
    clusterIDs = sch.fcluster(linkage, max_cluster_distance, 'distance')
    unique_clusterIDs = np.unique(clusterIDs)
    global_inds = np.array(list(range(len(features))))
    saved_inds = []

    # print('cluster IDs:', clusterIDs)

    for clusterID in unique_clusterIDs:
        cluster_features = features[np.where(clusterIDs == clusterID)]
        cluster_inds = global_inds[np.where(clusterIDs == clusterID)]
        if len(cluster_features) > 1:
            best_ind = cluster_inds[np.argmax(cluster_features[:, 6])]
            saved_inds.append(best_ind)
        else:
            saved_inds.append(cluster_inds[0])

    print('Reducing {} features to {}'.format(len(features), len(saved_inds)))

    return saved_inds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('encoder_config', type=str, help='Path to encoder config file')
    parser.add_argument('frames_dir', type=str, help='Path to the frames directory. Frames filename format: %06d.jpg')
    parser.add_argument('track_file', type=str, help='Path to the track results txt file with MOTChallenge format')
    parser.add_argument('feature_path', type=str, help='Path to the directory for writing feature files')
    parser.add_argument('--sample-rate', type=float, required=False, default=1.0 / 30.0,
                        help='Sample rate for detections')
    parser.add_argument('--score-threshold', type=float, required=False, default=0.8,
                        help='Minimum score of a qualified detection')
    parser.add_argument('--cluster-threshold', type=float, required=False, default=0.001,
                        help='Minimum similarity for clustering')
    args = parser.parse_args()

    assert os.path.isfile(args.encoder_config), 'Path to encoder config file \'{}\' is invalid'.format(
        args.encoder_config)
    assert os.path.isdir(args.frames_dir), 'Path to frame directory \'{}\' is invalid'.format(args.frames_dir)
    assert os.path.isfile(args.track_file), 'Path to tracking result file \'{}\'is invalid'.format(args.track_file)
    if not os.path.isdir(args.feature_path):
        os.makedirs(args.feature_path)

    tracks = np.loadtxt(args.track_file, delimiter=',')

    ids = np.unique(tracks[:, 1])
    print('All target IDs: ', ids)

    encoder = build_encoder(cfg_from_file(args.encoder_config))

    for target_id in ids:
        feature_filepath = os.path.join(args.feature_path, '{:06d}'.format(int(target_id)))
        detections = sample_detections(tracks, target_id, args.sample_rate, args.score_threshold)
        if len(detections) > 0:
            features = []
            for detection in detections:
                frame_id, _, l, t, w, h, score, _, _, _ = detection
                img = cv2.imread(os.path.join(args.frames_dir, '{:06d}.jpg'.format(int(frame_id))))
                feature = encoder([Detection(box=np.array([l, t, l + w, t + h]), score=score)], img)[0]  # shape=(1024,)
                features.append(feature)
            features = np.stack(features, axis=0)
            features = np.concatenate((detections, features), axis=1)  # shape=(N, 1024 + 10)
            if len(features) > 1:
                features = features[filter_features(features, max_cluster_distance=args.cluster_threshold)]
            np.save(feature_filepath, features)
            print('Saving {} data to {}'.format(features.shape, feature_filepath))
        else:
            raise AssertionError('No detections sampled for target id #{}'.format(target_id))

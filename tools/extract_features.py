import os
import cv2
import random
import argparse
import numpy as np
import scipy.io as sio
from typing import List
import scipy.cluster.hierarchy as sch

from mot.utils import cfg_from_file
from mot.encode import build_encoder
from mot.structures import Detection


def sample_detections(tracks: np.ndarray, target_id: int, sample_rate: float) -> np.ndarray:
    all_detections = tracks[np.where(tracks[:, 1] == target_id)]
    # inds = np.where(all_detections[:, 6] >= score_threshold)[0]
    # if not inds.any():
    #     print('Target {} has no confident detections. Fetching most confident detection with a score of {}'.format(
    #         target_id, all_detections[:, 6].max()))
    #     return [all_detections[np.argmax(all_detections[:, 6])]]
    #
    # all_detections = all_detections[inds]
    detections = []
    for detection in all_detections:
        if sample_rate > random.random():
            detections.append(detection)
    if len(detections) == 0:
        detections.append(all_detections[np.argmax(all_detections[:, 6])])
    print('Target #{} has {} detections and {} are sampled'.format(target_id, len(all_detections), len(detections)))
    return np.stack(detections, axis=0)


def filter_features(features: np.ndarray, max_cluster_distance: float) -> List[int]:
    M = 1 - (np.matmul(features, features.T) / np.square(np.linalg.norm(features, ord=2, axis=1, keepdims=True)).repeat(
        len(features), axis=1))

    iu = np.triu_indices(len(features), 1, len(features))
    M = M[iu]

    print('Min distance = ', M.min())
    print('Max distance = ', M.max())

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
    parser.add_argument('duke_path', type=str, help='Path to the DukeMTMC dataset')
    parser.add_argument('feature_path', type=str, help='Path to the directory for writing feature files')
    parser.add_argument('--sample-rate', type=float, required=False, default=1.0 / 30.0,
                        help='Sample rate for detections')
    # parser.add_argument('--score-threshold', type=float, required=False, default=0.8,
    #                     help='Minimum score of a qualified detection')
    parser.add_argument('--cluster-threshold', type=float, required=False, default=0.01,
                        help='Minimum similarity for clustering')
    args = parser.parse_args()

    assert os.path.isfile(args.encoder_config), 'Path to encoder config file \'{}\' is invalid'.format(
        args.encoder_config)
    if not os.path.isdir(args.feature_path):
        os.makedirs(args.feature_path)

    tracks = sio.loadmat(os.path.join(args.duke_path, 'ground_truth', 'saved.mat'))['trainData']

    for camID in range(1, 9):
        cam_feature_dir = os.path.join(args.feature_path, 'camera{:d}'.format(camID))
        if not os.path.isdir(cam_feature_dir):
            os.makedirs(cam_feature_dir)

        # From DukeMTMCC format (cam, ID, frameID, left, top, width, height, conf, x, y, z)
        # To MOTChallenge format (frameID, ID, left, top, width, height, conf, x, y, z)
        camData = tracks[np.where(tracks[:, 0] == camID)][:, 1:]
        camData[:, (0, 1)] = camData[:, (1, 0)]

        ids = np.unique(camData[:, 1]).astype(np.int)
        print('All target IDs in cam{}: '.format(camID), ids)

        encoder = build_encoder(cfg_from_file(args.encoder_config))

        global target_id
        for target_id in ids:
            feature_filepath = os.path.join(cam_feature_dir, '{:06d}'.format(int(target_id)))
            detections = sample_detections(camData, target_id, args.sample_rate)
            if len(detections) > 0:
                features = []
                for i, detection in enumerate(detections):
                    frame_id, _, l, t, w, h, score, _, _, _ = detection
                    img = cv2.imread(
                        os.path.join(args.duke_path, 'images/camera{}/{:06d}.jpg'.format(camID, int(frame_id))))

                    feature = encoder([Detection(box=np.array([l, t, l + w, t + h]), score=score)], img)[0]
                    features.append(feature)
                features = np.stack(features, axis=0)
                # features = np.concatenate((detections, features), axis=1)  # shape=(N, 1024 + 10)
                if len(features) > 1:
                    filtered_inds = filter_features(features, max_cluster_distance=args.cluster_threshold)
                    detections = detections[filtered_inds]
                    features = features[filtered_inds]
                data = np.concatenate((detections, features), axis=1)
                np.save(feature_filepath, data)
                print('Saving {} data to {}'.format(data.shape, feature_filepath))
            else:
                raise AssertionError('No detections sampled for target id #{}'.format(target_id))

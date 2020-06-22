import os
import cv2
import sys
import torch
import argparse
import numpy as np

sys.path.append('/home/linkinpark213/Source/online-mot-by-detection')
from mot.encode import build_encoder
from mot.utils.config import cfg_from_file
from mot.structures import Detection

sys.path.append('/home/linkinpark213/Source/online-mot-by-detection/mot/encode/OpenReID/')
from reid.evaluators import pairwise_distance, evaluate_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to dataset')
    args = parser.parse_args()

    encoder_cfg = cfg_from_file('/home/linkinpark213/Source/online-mot-by-detection/configs/encode/openreid.py')
    print('Building encoder...')
    encoder = build_encoder(encoder_cfg)

    subset_features = {}

    for subset in ['query', 'gallery']:
        ids = sorted(os.listdir(os.path.join(args.dataset_path, subset)))

        all_infos = []
        all_features = []

        for target_id in ids:
            image_filenames = os.listdir(os.path.join(args.dataset_path, subset, target_id))
            for image_filename in image_filenames:
                print('Encoding {}'.format(os.path.join(args.dataset_path, subset, target_id, image_filename)))
                cam_id, frame_id = int(image_filename.split('_')[1][1]), int(
                    image_filename.split('.')[0].split('_')[2][1:])
                image = cv2.imread(os.path.join(args.dataset_path, subset, target_id, image_filename))
                h, w, _ = image.shape
                feature = encoder([Detection(np.array([0, 0, w, h]), score=1)], image)[0]
                all_infos.append([int(target_id), cam_id, frame_id])
                all_features.append(feature)

        all_infos = np.array(all_infos)
        all_features = np.hstack((all_infos, all_features))
        subset_features[subset] = all_features

    query_features, gallery_features = subset_features['query'], subset_features['gallery']

    distmat = pairwise_distance(torch.tensor(query_features[:, 3:]), torch.tensor(gallery_features[:, 3:]))

    query = query_features[:, (2, 0, 1)]
    gallery = gallery_features[:, (2, 0, 1)]

    print('Shape of distance matrix: ', distmat.shape)

    evaluate_all(distmat, query, gallery)

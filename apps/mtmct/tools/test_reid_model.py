import os
import sys
import cv2
import argparse
import numpy as np

sys.path.append('/home/linkinpark213/Source/online-mot-by-detection')
from mot.encode import build_encoder
from mot.utils.config import cfg_from_file
from mot.structures import Detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1_path', type=str, help='Path to image 1')
    parser.add_argument('img2_path', type=str, help='Path to image 2')
    args = parser.parse_args()

    encoder_cfg = cfg_from_file('/home/linkinpark213/Source/online-mot-by-detection/configs/encode/openreid_r50.py')
    print('Building encoder...')
    encoder = build_encoder(encoder_cfg)

    print('Encoding...')
    images = cv2.imread(args.img1_path), cv2.imread(args.img2_path)
    features = []
    for i in range(2):
        h, w, _ = images[i].shape
        features.extend(encoder([Detection(np.array([0, 0, w, h]), score=1)], images[i]))

    print('Feature norms = {}, {}'.format(np.linalg.norm(features[0], ord=2), np.linalg.norm(features[1], ord=2)))
    print('Feature distance = {}'.format(1 - np.matmul(features[0], features[1].T)))

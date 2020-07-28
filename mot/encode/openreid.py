import os
import cv2
import sys
import torch
import numpy as np
from typing import List, Tuple
import torchvision.transforms as T
from torch.autograd import Variable

from .encode import ENCODER_REGISTRY, Encoder
from mot.structures import Detection

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../third_party', 'OpenReID'))
import reid.models
from reid.utils.get_loaders import checkpoint_loader
from reid.feature_extraction.cnn import extract_cnn_feature

__all__ = ['OpenReIDEncoder']

import time


@ENCODER_REGISTRY.register()
class OpenReIDEncoder(Encoder):
    def __init__(self, model_name: str, checkpoint_path: str, arch: str = 'resnet50',
                 img_size: Tuple[int, int] = (128, 256), norm: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.model = reid.models.create(model_name, feature_dim=256, norm=norm, num_classes=0, last_stride=2,
                                        arch=arch)

        self.model, start_epoch, best_top1 = reid.utils.get_loaders.checkpoint_loader(self.model, checkpoint_path)
        self.model = self.model.eval().cuda()
        self.size = img_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.transform(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def encode(self, detections: List[Detection], full_img: np.ndarray) -> List[object]:
        if len(detections) > 0:
            all_crops = []
            full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
            for detection in detections:
                box = detection.box
                crop = full_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if crop.shape[0] * crop.shape[1] > 0:
                    all_crops.append(crop)
                else:
                    all_crops.append(np.ones((10, 10, 3)).astype(np.float32) * 255)

            im_batch = self._preprocess(all_crops)
            input_img = Variable(im_batch.cuda())

            start_time = time.time()
            outputs = extract_cnn_feature(self.model, input_img)
            end_time = time.time()

            print('Encoder spent {}ms'.format(1000 * (end_time - start_time)))


            return outputs.detach().numpy()
        else:
            return []

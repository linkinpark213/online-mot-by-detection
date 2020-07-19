import cv2
import torch
import numpy as np
from typing import List, Tuple
from torch2trt import TRTModule
import torchvision.transforms as T

from mot.encode import Encoder, ENCODER_REGISTRY
from mot.structures import Detection


@ENCODER_REGISTRY.register()
class TRTOpenReIDEncoder(Encoder):
    def __init__(self, trt_checkpoint_path: str, img_size: Tuple[int, int] = (128, 256), max_batch_size: int = 8,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(trt_checkpoint_path))
        self.model_trt = self.model_trt.cuda().eval()
        self.size = img_size
        self.max_batch_size = max_batch_size
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

            outputs = []
            for i in range(0, len(all_crops), self.max_batch_size):
                im_batch = self._preprocess(all_crops[i: min(len(all_crops), i + self.max_batch_size)])
                im_batch = im_batch.cuda()
                output = self.model_trt(im_batch)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            return outputs.cpu().detach().numpy()
        else:
            return []

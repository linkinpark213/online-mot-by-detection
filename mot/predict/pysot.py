# Part of this code is borrowed from official implementation of SiameseRPN in PySOT library:
# https://github.com/STVIR/pysot/blob/master/pysot/tracker/siamrpn_tracker.py

import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from pysot.utils.anchor import Anchors
from pysot.core.config import cfg as pysot_cfg
from pysot.models.model_builder import ModelBuilder

from mot.utils import crop
from mot.structures import Tracklet, Prediction
from .predict import Predictor, PREDICTOR_REGISTRY


@PREDICTOR_REGISTRY.register()
class SiamRPNPredictor(Predictor):
    def __init__(self, config: str, checkpoint: str, x_size: Tuple[int, int] = (256, 256), max_batch_size: int = 20,
                 **kwargs):
        super(SiamRPNPredictor).__init__()

        self.pysot_cfg = pysot_cfg
        self.pysot_cfg.merge_from_file(config)

        self.model = ModelBuilder()
        self.model.load_state_dict(torch.load(checkpoint,
                                              map_location=lambda storage, loc: storage.cpu()))
        self.model.eval().cuda()

        self.zf = None
        self.x_size = x_size
        self.max_batch_size = max_batch_size
        self.anchor_num = len(self.pysot_cfg.ANCHOR.RATIOS) * len(self.pysot_cfg.ANCHOR.SCALES)
        self.score_size = (self.pysot_cfg.TRACK.INSTANCE_SIZE - self.pysot_cfg.TRACK.EXEMPLAR_SIZE) // \
                          self.pysot_cfg.ANCHOR.STRIDE + 1 + self.pysot_cfg.TRACK.BASE_SIZE
        self.anchors = self.generate_anchor(self.score_size)

    def initiate(self, tracklets: List[Tracklet]) -> None:
        pass

    def update(self, tracklets: List[Tracklet]) -> None:
        assert len(tracklets) == 0 or 'patch' in tracklets[
            0].feature.keys(), 'An ImagePatchEncoder named `patch` should be enabled to run single object tracking'

        z = torch.tensor(np.stack([tracklet.feature['patch'] for tracklet in tracklets]).transpose(0, 3, 1, 2)).type(
            torch.float32).cuda()
        self.zf = self.model.backbone(z)
        if self.pysot_cfg.ADJUST.ADJUST:
            self.zf = self.model.neck(self.zf)

    def generate_anchor(self, score_size):
        anchors = Anchors(self.pysot_cfg.ANCHOR.STRIDE,
                          self.pysot_cfg.ANCHOR.RATIOS,
                          self.pysot_cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _predict(self, tracklets: List[Tracklet], img: np.ndarray, ind_range: Tuple[int, int]) -> List[Prediction]:
        predictions = []
        xboxes = np.stack([tracklet.last_detection.box for tracklet in tracklets])
        xboxes[:, 2] = xboxes[:, 2] - xboxes[:, 0]
        xboxes[:, 3] = xboxes[:, 3] - xboxes[:, 1]
        xboxes[:, 0] = xboxes[:, 0] + xboxes[:, 2] / 2
        xboxes[:, 1] = xboxes[:, 1] + xboxes[:, 3] / 2
        xboxes[:, 2] = xboxes[:, 2] * 2
        xboxes[:, 3] = xboxes[:, 3] * 2

        x = np.stack(
            [crop(img, xbox[0], xbox[1], (xbox[2], xbox[3]), resize_to=self.x_size) for xbox in xboxes]).transpose(
            0, 3, 1, 2)
        resize_ratios = [min(self.x_size[0] / xbox[2], self.x_size[1] / xbox[3]) for xbox in xboxes]
        x = torch.tensor(x).type(torch.float32).cuda()
        xf = self.model.backbone(x)
        if self.pysot_cfg.ADJUST.ADJUST:
            xf = self.model.neck(xf)

        cls, loc = self.model.rpn_head(self.zf[ind_range[0]:ind_range[1]], xf[ind_range[0]:ind_range[1]])

        for i in range(len(tracklets)):
            score = self._convert_score(cls[i:i + 1])
            pred_bbox = self._convert_bbox(loc[i:i + 1], self.anchors)

            pred_ind = np.argmax(score)
            pred_bbox = pred_bbox[:, pred_ind]
            pred_bbox = pred_bbox / resize_ratios[i]
            pred_bbox[0] += xboxes[i][0] - pred_bbox[2] / 2
            pred_bbox[1] += xboxes[i][1] - pred_bbox[3] / 2
            pred_bbox[2] += pred_bbox[0]
            pred_bbox[3] += pred_bbox[1]

            predictions.append(Prediction(pred_bbox, score))

        del cls, loc
        return predictions

    def predict(self, tracklets: List[Tracklet], img: np.ndarray) -> List[Prediction]:
        predictions = []
        if len(tracklets) > 0:
            self.update(tracklets)
            if len(tracklets) > self.max_batch_size:
                # Since target number is uncertain, a large number of targets may cause an OOM during SOT
                for i in range(len(tracklets), self.max_batch_size):
                    r = (i, min(len(tracklets), i + self.max_batch_size))
                    predictions.extend(self._predict(tracklets, img, r))
            else:
                predictions.extend(self._predict(tracklets, img, (0, len(tracklets))))
        return predictions

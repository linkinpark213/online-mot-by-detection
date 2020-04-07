import mmcv
import torch
import numpy as np
from typing import List
from mot.detect import MMDetector
from mmdet.apis import init_detector
from mmdet.models import TwoStageDetector
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.core import bbox2result, bbox2roi

from mot.structures import Tracklet, Prediction
from .predict import Predictor, PREDICTOR_REGISTRY


@PREDICTOR_REGISTRY.register()
class MMTwoStagePredictor(Predictor):
    """
    By far this is only tested on Faster R-CNN, but should work for all TwoStageDetectors.
    So Cascade R-CNN isn't supported.
    """

    def __init__(self, config: str, checkpoint: str, conf_threshold: float = 0.5, **kwargs):
        super(MMTwoStagePredictor).__init__()
        self.model = init_detector(
            config,
            checkpoint,
            device=torch.device('cuda', 0))
        self.conf_thres = conf_threshold

    def initiate(self, tracklets: List[Tracklet]) -> None:
        # No need to initiate
        pass

    def update(self, tracklets: List[Tracklet]) -> None:
        # No need to update
        pass

    def regress_and_classify(self, img, tracklets):
        cfg = self.model.cfg
        device = next(self.model.parameters()).device

        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)

        data = dict(img=img)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        proposals = np.array(
            [[*(tracklet.last_detection.box[:]), tracklet.last_detection.score] for tracklet in tracklets])
        proposals[:, 0:4] = proposals[:, 0:4] * data['img_meta'][0][0]['scale_factor']
        proposals_tensor = torch.tensor(proposals).cuda(device)

        with torch.no_grad():
            x = self.model.extract_feat(data['img'][0])

            # Here I re-implement BBoxTestMixin.simple_test_bboxes().
            # It seems that the bboxes shouldn't have been rescaled when calling get_det_bboxes()?
            rois = bbox2roi([proposals_tensor])
            roi_feats = self.model.bbox_roi_extractor(
                x[:len(self.model.bbox_roi_extractor.featmap_strides)], rois)
            if self.model.with_shared_head:
                roi_feats = self.model.shared_head(roi_feats)
            cls_score, bbox_pred = self.model.bbox_head(roi_feats)
            bboxes, scores = self.model.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                data['img_meta'][0][0]['img_shape'],
                data['img_meta'][0][0]['scale_factor'],
                # I mean here
                rescale=True,
                cfg=None)

            bboxes = bboxes.view(-1, 81, 4)
            bboxes = torch.cat((bboxes[:, 1, :], scores[:, 1:2]), dim=1)
            labels = torch.tensor([0] * bboxes.shape[0])
            bbox_results = bbox2result(bboxes, labels, self.model.bbox_head.num_classes)[0]

            return bbox_results

    def predict(self, tracklets: List[Tracklet], img: np.ndarray) -> List[Prediction]:
        if len(tracklets) != 0:
            bbox_results = self.regress_and_classify(img, tracklets)
            predictions = []
            for i, tracklet in enumerate(tracklets):
                tracklet.prediction = Prediction(bbox_results[i, 0:4], bbox_results[i, 4])
                predictions.append(tracklet.prediction)
            return predictions
        else:
            return []


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

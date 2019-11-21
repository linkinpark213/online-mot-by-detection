import torch
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from .predict import Predictor, Prediction
from detectron2.structures import Boxes, Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model, detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs


class DetectronRCNNPredictor(Predictor):
    def __init__(self, config, checkpoint=None, conf_threshold=0.5):
        super(DetectronRCNNPredictor).__init__()
        cfg = get_cfg()
        cfg.merge_from_file(config)
        if checkpoint is not None:
            cfg.MODEL.WEIGHTS = checkpoint
        self.model = build_model(cfg)
        self.model.proposal_generator = None
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.conf_threshold = conf_threshold

    def __call__(self, tracklets, img):
        return self.predict(tracklets, img)

    def initiate(self, tracklets):
        # No need to initiate
        pass

    def update(self, tracklets):
        # No need to update
        pass

    def regress_and_classify(self, image, tracklets):
        # Convert boxes to proposals
        height, width = image.shape[:2]
        image = self.transform_gen.get_transform(image).apply_image(image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        proposal_boxes = Boxes(torch.tensor([tracklet.last_detection.box for tracklet in tracklets]))
        proposals = Instances((height, width), proposal_boxes=proposal_boxes)

        inputs = {"image": image, "height": height, "width": width, "proposals": proposals}

        images = self.model.preprocess_image([inputs])
        features = self.model.backbone(images.tensor)
        proposals = [inputs["proposals"].to(self.model.device)]

        # Extract features, perform RoI pooling and perform regression/classification for each RoI
        features_list = [features[f] for f in self.model.roi_heads.in_features]

        box_features = self.model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.model.roi_heads.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.model.roi_heads.box_predictor(box_features)
        del box_features

        raw_outputs = FastRCNNOutputs(
            self.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.model.roi_heads.smooth_l1_beta,
        )

        # Convert raw outputs to predicted boxes and scores
        boxes = raw_outputs.predict_boxes()[0]
        scores = raw_outputs.predict_probs()[0]

        num_bbox_reg_classes = boxes.shape[1] // 4
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip((height, width))
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)
        boxes = boxes[:, 0, :]
        scores = scores[:, 0]

        pred_boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        return pred_boxes, scores

    def predict(self, tracklets, img):
        # Use last detection boxes of tracklets as region proposals
        if len(tracklets) != 0:
            pred_boxes, scores = self.regress_and_classify(img, tracklets)

            predictions = []
            for i, tracklet in enumerate(tracklets):
                tracklet.prediction = Prediction(pred_boxes[i], scores[i])
                predictions.append(tracklet.prediction)
            return predictions
        else:
            return []

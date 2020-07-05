import os
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import List
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../third_party', 'PCB'))
from model import PCB, PCB_test

from mot.structures import Detection
from .encode import Encoder, ENCODER_REGISTRY


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.ones(n, n).cuda()
        dist.addmm_(1, -1, inputs, inputs.t())
        mask = targets.expand(n, n).eq(targets.expand(n, n).t()).float()
        mask = mask + torch.eye(n).cuda() * 0.5

        dist_ap, dist_an = [], []
        for i in range(n):
            positive = dist[i][mask[i] == 1]
            negative = dist[i][mask[i] == 0]
            min_sample = min(len(positive), len(negative))
            min_sample = min(min_sample, 5)
            positive = random.sample(positive.cpu().detach().tolist(), min_sample)
            negative = random.sample(negative.cpu().detach().tolist(), min_sample)
            dist_ap += positive
            dist_an += negative
        dist_ap = torch.Tensor(dist_ap)
        dist_an = torch.Tensor(dist_an)

        # Compute ranking hinge loss
        y = torch.ones(len(dist_an))
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        aaa = (dist_an.data > dist_ap.data).sum() * 1.
        prec = aaa.numpy() / y.size(0)
        return loss, prec


@ENCODER_REGISTRY.register()
class PCBEncoder(Encoder):
    def __init__(self, model_path: str, name: str = 'pcb', **kwargs):
        super(PCBEncoder).__init__()
        self.name = name
        model_structure = PCB(751)
        model = model_structure.convert_to_rpp()
        save_path = os.path.join(model_path)
        model.load_state_dict(torch.load(save_path))

        self.model = PCB_test(model, True)
        self.size = (64, 128)
        self.feature_H = True
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # settings
        ignored_params = list(map(id, self.model.classifiers.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

        self.optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.0001},
            {'params': self.model.classifiers.parameters(), 'lr': 0.0001},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=40, gamma=0.1)
        self.criterion = TripletLoss()

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def learn_online(self, feature_matrix, labels):
        if feature_matrix:
            self.model.train(True)
            inputs = torch.Tensor(feature_matrix)
            labels = torch.Tensor(labels)
            inputs = Variable(inputs.cuda(), requires_grad=True)
            labels = Variable(labels.cuda())
            self.optimizer_ft.zero_grad()
            loss, prec = self.criterion(inputs, labels)
            loss.requires_grad_()
            loss.backward()
            self.scheduler.step()
            print('Loss: {:.4f} Acc: {:.4f}'.format(loss, prec))
        else:
            print('Nothing to learn')

    # Extract feature
    def fliplr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def encode(self, detections: List[Detection], full_img: np.ndarray) -> List[object]:
        model = self.model.eval()
        model = model.cuda()
        features = torch.FloatTensor()
        all_crops = []
        for detection in detections:
            box = detection.box
            crop = full_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if crop.shape[0] * crop.shape[1] > 0:
                all_crops.append(crop)
            else:
                all_crops.append(np.ones((10, 10, 3)).astype(np.float32) * 255)
        if detections.shape[0] != 0:
            img = self._preprocess(all_crops)
            n, c, h, w = img.shape
            if self.feature_H:
                ff = torch.FloatTensor(n, 256, 6).zero_()  # we have six parts
            else:
                ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts
            for i in range(2):
                if (i == 1):
                    img = self.fliplr(img)
                input_img = Variable(img.cuda())
                outputs = model(input_img)
                f = outputs.data.cpu()
                ff = ff + f
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            return torch.cat((features, ff), 0)
        else:
            return torch.zeros(1)

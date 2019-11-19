import os
import cv2
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from .encode import Encoder
from .PCB.loss import TripletLoss
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from .DGNet.reIDmodel import ft_net, ft_netAB, ft_net_dense, PCB, PCB_test
import pdb


class DGNetEncoder(Encoder):
    def __init__(self, model_path, name='dgnet'):
        super(DGNetEncoder).__init__()
        self.name = name

        self.model = ft_netAB(751, norm=False, stride=1, pool='max')
        save_path = os.path.join(model_path, 'id_00100000.pt')
        state_dict = torch.load(save_path)
        self.model.load_state_dict(state_dict['a'], strict=False)
        self.model.classifier1.classifier = nn.Sequential()
        self.model.classifier2.classifier = nn.Sequential()
        self.model = self.model.eval().cuda()
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        """
        # settings
        ignored_params = list(map(id, self.model.classifiers.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

        self.optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.0001},
            {'params': self.model.classifiers.parameters(), 'lr': 0.0001},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        self.scheduler = lr_scheduler.StepLR(self.optimizer_ft, step_size=40, gamma=0.1)
        self.criterion = TripletLoss()
        """

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

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def normlize(self, f):
        # f = f.squeeze()
        fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
        f = f.div(fnorm.expand_as(f))
        return f

    def __call__(self, detections, full_img):
        features = torch.FloatTensor()
        all_crops = []
        for detection in detections:
            box = detection.box
            crop = full_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if crop.shape[0] * crop.shape[1] > 0:
                all_crops.append(crop)
            else:
                all_crops.append(np.ones((10, 10, 3)).astype(np.float32) * 255)
        if len(detections) != 0:
            img = self._preprocess(all_crops)
            n, c, h, w = img.shape
            ff = torch.FloatTensor(n, 1024).zero_()
            for i in range(2):
                if (i == 1):
                    img = self.fliplr(img)
                input_img = Variable(img.cuda())
                f, x = self.model(input_img)
                x[0] = self.normlize(x[0])
                x[1] = self.normlize(x[1])
                f = torch.cat((x[0], x[1]), dim=1)  # use 512-dim feature
                f = f.data.cpu()
                ff = ff + f

            ff[:, 0:512] = self.normlize(ff[:, 0:512])
            ff[:, 512:1024] = self.normlize(ff[:, 512:1024])
            ff[:, 512:1024] = ff[:, 512:1024] * 0.7
            return torch.cat((features, ff), 0)
        else:
            return torch.zeros(1)

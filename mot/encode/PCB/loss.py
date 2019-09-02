import torch
from torch import nn
from torch.autograd import Variable
import pdb
import random


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

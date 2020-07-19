import os
import sys
import torch
import traceback
import numpy as np
from torch2trt import torch2trt
import torch.nn.functional as F
from torch2trt import TRTModule

sys.path.append('/home/ubilab/Source/open-reid-tracking')
import reid.models
from reid.utils.get_loaders import checkpoint_loader


# Replace forward function with only a feature vector output and removed procedures for training
def forward_eval(self, x):
    x = self.base(x)
    x = self.global_avg_pool(x).view(x.shape[0], -1)
    x = self.feat_fc(x)

    feature_ide = x

    feature_ide = F.normalize(feature_ide)

    return feature_ide


# Check the output against PyTorch
def diff(x, model1, model2):
    y = model1(x)
    print(y)
    y_trt = model2(x)
    print(y_trt)
    return torch.max(torch.abs(y - y_trt))


if __name__ == '__main__':
    # Load model
    model = reid.models.create('ide', feature_dim=256, norm=True, num_classes=0, last_stride=2, arch='resnet50')
    model, epoch, best_top1 = checkpoint_loader(model,
                                                '/home/ubilab/Source/open-reid-tracking/logs/zju/duke_reid/2020-06-11_17-13-49/model_best.pth.tar')
    model = model.eval().cuda()

    model.forward = forward_eval.__get__(model, reid.models.IDE_model)

    # Dummy input data
    x = np.random.random([8, 3, 256, 128]).astype(np.float32)
    x = torch.tensor(x).cuda()

    # Convert to TensorRT model
    model_trt = torch2trt(model, [x], max_batch_size=8)

    # Max difference between PyTorch model and converted TRT model
    print('Max difference between PyTorch model and converted TRT model', diff(x, model, model_trt).data)

    # Save TensorRT model
    torch.save(model_trt.state_dict(), 'openreid_ide_r50_trt.pth')
    print('Model saved to openreid_ide_r50_trt.pth .')

    # Load TensorRT model again
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('openreid_ide_r50_trt.pth'))
    model_trt = model_trt.eval().cuda()

    # Max difference between PyTorch model and loaded TRT model
    print('Max difference between PyTorch model and loaded TRT model', diff(x, model, model_trt).data)

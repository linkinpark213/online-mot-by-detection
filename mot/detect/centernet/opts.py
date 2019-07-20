from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class opts(object):
    def __init__(self):
        self.task = 'ctdet'
        self.load_model = 'mot/detect/centernet/models/ctdet_coco_dla_2x.pth'
        self.gpus = [0]
        self.num_workers = 4
        self.arch = 'dla_34'
        self.heads = {
            'hm': 80,
            'wh': 2,
            'reg': 2
        }
        self.head_conv = -1

        self.down_ratio = 4
        self.input_res = -1
        self.input_h = -1
        self.input_w = -1
        self.test_scales = [1]
        self.K = 100
        self.reg_loss = 'l1'
        self.hm_weight = 1
        self.off_weight = 1
        self.wh_weight = 0.1
        self.mean = [0.408, 0.447, 0.470]
        self.std = [0.289, 0.274, 0.278]
        self.dataset = 'coco'
        self.num_classes = 80
        self.debug = False
        self.debugger_theme = 'white'
        self.fix_res = False
        self.pad = 31
        self.flip_test = False
        self.reg_offset = False
        self.nms = False

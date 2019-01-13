# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 15:41
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : ordinalnet.py
# @Software: PyCharm

import os
import math
import torch as t
from warnings import warn
from torch import nn
from torchvision.models import alexnet, vgg
from torch.nn import functional as F
from core.config import Config
from core.models import _BaseModule


class OrdinalNet(_BaseModule):
    def __init__(self, config: Config):
        super(OrdinalNet, self).__init__()
        if config.use_batch_norm:
            self.vgg_c = vgg.vgg11_bn(num_classes=config.num_classes)
            self.vgg_r = vgg.vgg11_bn()
        else:
            self.vgg_c = vgg.vgg11(config.use_pytorch_weight)
            self.vgg_r = vgg.vgg11(config.use_pytorch_weight)
        self.alex = alexnet(num_classes=config.num_classes)
        self.regression = nn.Linear(1000, 1)

    def forward(self, tensor_p, tensor_s):
        # out = F.softmax(self.vgg_c(input), dim=0)
        # out = self.logits(out)
        return prob_c, regr, prob_final

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

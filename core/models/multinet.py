# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 15:41
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : multinet.py
# @Software: PyCharm

import math
import torch as t
from torch import nn
from torchvision.models import resnet
from torch.nn import functional as F
from ..config import Config
from ..models import vgg
import _base


class MultiNet(_base._BaseModule):
    def __init__(self, config: Config):
        super(MultiNet, self).__init__()
        self.input_shape = (-1, 3, config.image_resize[0], config.image_resize[1])
        if config.use_batch_norm:
            self.vgg_c = vgg.vgg11_bn(num_classes=config.num_classes, init_weights=False)
            self.vgg_r = vgg.vgg11_bn(init_weights=False)
        else:
            self.vgg_c = vgg.vgg11(num_classes=config.num_classes, init_weights=False)
            self.vgg_r = vgg.vgg11(init_weights=False)
        self.regression = nn.Linear(1000, 1)
        self.resnet = resnet.resnet18(num_classes=config.num_classes)

    def forward(self, tensor_s, tensor_p):
        tensor_s = tensor_s.view(self.input_shape)
        mini_batch = tensor_s.size(0)
        sub_features, interim_probs = self.vgg_c(tensor_s)
        pare_features, para_logits = self.vgg_r(tensor_p)
        cover_rate = self.regression(F.dropout(F.relu(para_logits)))
        sub_features = t.split(sub_features, 8, dim=0)
        pare_features = t.split(pare_features, 1, dim=0)
        features = []
        for sub, pare in zip(sub_features, pare_features):
            pare = pare.expand(8, -1, -1, -1)
            features.append(t.cat((sub, pare), dim=1))
        features = t.cat(features).view(mini_batch, 1, 224, 224)  # shape: [mini_batch, 1, 224,224]
        features = features.expand(-1, 3, -1, -1)
        final_probs = self.resnet(features)
        return interim_probs, cover_rate, final_probs

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

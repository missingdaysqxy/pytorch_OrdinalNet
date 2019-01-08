# -*- coding: utf-8 -*-
# @Time    : 2019/1/8/008 9:06 上午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : ordinalloss.py
# @Software: PyCharm

import torch as t
from torch import nn
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import functional as F
import math


class OrdinalLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(OrdinalLoss,self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input,target):
        raise RuntimeError
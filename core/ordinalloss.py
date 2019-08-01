# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 9:06
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : ordinalloss.py
# @Software: PyCharm

import torch as t
from torch import nn, autograd
from torch.nn import Parameter as P
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import functional as F
import math

import ipdb


def ordinal_loss(input: t.Tensor, target: t.Tensor, MaxValue):
    if (t.argsort(input) == t.argsort(target)).all():
        return 0
    else:
        in_padL = F.pad(input, [1, 0], mode='constant', value=input[-1].data)
        in_padR = F.pad(input, [0, 1], mode='constant', value=input[0].data)
        in_diff = in_padR - in_padL
        tar_padL = F.pad(target, [1, 0], mode='constant', value=target[-1].data)
        tar_padR = F.pad(target, [0, 1], mode='constant', value=target[0].data)
        tar_diff = tar_padR - tar_padL
        loss = F.mse_loss(in_diff / MaxValue, tar_diff / MaxValue)
        return loss


class OrdinalLoss(_WeightedLoss):
    def __init__(self, MaxValue=1, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(OrdinalLoss, self).__init__(weight, size_average, reduce, reduction)
        assert MaxValue != 0
        self.MaxVaule = MaxValue
        self.w = P(t.Tensor(2))

    def forward(self, input: t.Tensor, target: t.Tensor):
        return F.binary_cross_entropy(input, target) + ordinal_loss(input, target, self.MaxVaule)
        # predict = t.argmax(input, dim=-1)
        # correct_idx = t.nonzero(predict == target).squeeze_()
        # wrong_idx = t.nonzero(predict != target).squeeze_()
        # # compare in wrong_idx
        # w_score = 0
        # for i in range(wrong_idx.shape[0]):
        #     for j in range(i + 1, wrong_idx.shape[0]):
        #         target_ordinal = target[wrong_idx[j]] - target[wrong_idx[i]]
        #         predict_ordinal = predict[wrong_idx[j]] - predict[wrong_idx[i]]
        #         w_score += target_ordinal * predict_ordinal
        # # compare between correct_idx and wrong_idx
        # b_score = 0
        # for i in range(correct_idx.shape[0]):
        #     for j in range(wrong_idx.shape[0]):
        #         target_ordinal = target[correct_idx[j]] - target[wrong_idx[i]]
        #         predict_ordinal = predict[correct_idx[j]] - predict[wrong_idx[i]]
        #         b_score += target_ordinal * predict_ordinal
        # score = w_score + b_score
        # return math.exp(0 - self.w * score) + F.mseloss(target, input)


if __name__ == "__main__":
    loss_fn = OrdinalLoss()
    x1 = F.softmax(t.randint(1, 6, [5],dtype=t.float,requires_grad=True))
    x2 = F.softmax(t.randint_like(x1, 1, 6))
    print(x1, x2)
    y = loss_fn(x1, x2)
    print(y.requires_grad)
    print(y, y.grad)
    y.backward()
    print(y.grad)

    # x = t.ones((5, 3), requires_grad=True)
    # x = x.view((-1, 1))
    # w = t.randn(2, 15)
    # b = t.rand(2, 1)
    # y = t.mm(w, x) + b
    # print(y.requires_grad)
    # y.requires_grad_()
    # print(y.requires_grad)
    # print(y)
    # print(w, b)
    # print(y.grad)
    # y.backward()
    # print(y.grad)
    # print(w, b)

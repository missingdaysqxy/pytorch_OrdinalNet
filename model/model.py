# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 15:41
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : model.py
# @Software: PyCharm

import torch as t
from torch import nn
from torchvision.models import vgg
from torch.nn import functional as F
import math


class OrdinalNet(nn.Module):
    def __init__(self, num_classes, batch_norm=True, pretrained=False):
        super(OrdinalNet, self).__init__()
        if batch_norm:
            self.vggnet = vgg.vgg19_bn(pretrained)
        else:
            self.vggnet = vgg.vgg19(pretrained)
        # self.vggnet = MyVGG19(3, 1000, batch_norm)
        self.logits = nn.Linear(1000, num_classes)

    def forward(self, input):
        out = F.softmax(self.vggnet(input), dim=0)
        out = self.logits(out)
        return out

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


class MyVGG19(nn.Module):
    def __init__(self, in_channels, num_classes, batch_norm=False):
        super(MyVGG19, self).__init__()
        self.batch_norm = batch_norm
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn3_4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn4_4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5_4 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        # self.bn6 = nn.BatchNorm2d(4096)
        self.fc7 = nn.Linear(4096, 4096)
        # self.bn7 = nn.BatchNorm2d(4096)
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, input):
        if self.batch_norm:
            out = F.relu(self.bn1_1(self.conv1_1(input)))
            out = F.relu(self.bn1_2(self.conv1_2(out)))
            out = self.pool1(out)

            out = F.relu(self.bn2_1(self.conv2_1(out)))
            out = F.relu(self.bn2_2(self.conv2_2(out)))
            out = self.pool2(out)

            out = F.relu(self.bn3_1(self.conv3_1(out)))
            out = F.relu(self.bn3_2(self.conv3_2(out)))
            out = F.relu(self.bn3_3(self.conv3_3(out)))
            out = F.relu(self.bn3_4(self.conv3_4(out)))
            out = self.pool3(out)

            out = F.relu(self.bn4_1(self.conv4_1(out)))
            out = F.relu(self.bn4_2(self.conv4_2(out)))
            out = F.relu(self.bn4_3(self.conv4_3(out)))
            out = F.relu(self.bn4_4(self.conv4_4(out)))
            out = self.pool4(out)

            out = F.relu(self.bn5_1(self.conv5_1(out)))
            out = F.relu(self.bn5_2(self.conv5_2(out)))
            out = F.relu(self.bn5_3(self.conv5_3(out)))
            out = F.relu(self.bn5_4(self.conv5_4(out)))
            out = self.pool5(out)

            out = out.view(out.size(0), -1)

            out = F.relu(self.fc6(out))
            out = F.dropout(out)
            out = F.relu(self.fc7(out))
            out = F.dropout(out)
            out = self.fc8(out)
        else:
            out = F.relu(self.conv1_1(input))
            out = F.relu(self.conv1_2(out))
            out = self.pool1(out)

            out = F.relu(self.conv2_1(out))
            out = F.relu(self.conv2_2(out))
            out = self.pool2(out)

            out = F.relu(self.conv3_1(out))
            out = F.relu(self.conv3_2(out))
            out = F.relu(self.conv3_3(out))
            out = F.relu(self.conv3_4(out))
            out = self.pool3(out)

            out = F.relu(self.conv4_1(out))
            out = F.relu(self.conv4_2(out))
            out = F.relu(self.conv4_3(out))
            out = F.relu(self.conv4_4(out))
            out = self.pool4(out)

            out = F.relu(self.conv5_1(out))
            out = F.relu(self.conv5_2(out))
            out = F.relu(self.conv5_3(out))
            out = F.relu(self.conv5_4(out))
            out = self.pool5(out)

            out = out.view(out.size(0), -1)

            out = F.relu(self.fc6(out))
            out = F.dropout(out)
            out = F.relu(self.fc7(out))
            out = F.dropout(out)
            out = self.fc8(out)
        return out


'''
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace)
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace)
    (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace)
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace)
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace)
    (23): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (25): ReLU(inplace)
    (26): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (27): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace)
    (33): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (34): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (35): ReLU(inplace)
    (36): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (37): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (38): ReLU(inplace)
    (39): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace)
    (43): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (44): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (45): ReLU(inplace)
    (46): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (47): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (48): ReLU(inplace)
    (49): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (50): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (51): ReLU(inplace)
    (52): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=6, bias=True)
  )
)

'''

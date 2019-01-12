# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 15:41
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : models.py
# @Software: PyCharm

import os
import math
import torch as t
from warnings import warn
from torch import nn
from torchvision.models import vgg
from torch.nn import functional as F
from model.config import Config


class _BaseModule(nn.Module):
    def __int__(self, config: Config):
        super(_BaseModule, self).__init__()
        self.config = config

    def forward(self, input):
        raise NotImplementedError("Should be overridden by all subclasses.")

    def initialize_weights(self):
        raise NotImplementedError("Should be overridden by all subclasses.")


class OrdinalNet(_BaseModule):
    def __init__(self, config: Config):
        super(OrdinalNet, self).__init__()
        if config.use_batch_norm:
            self.vggnet = vgg.vgg11_bn(config.use_pytorch_weight)
        else:
            self.vggnet = vgg.vgg11(config.use_pytorch_weight)
        # self.vggnet = MyVGG19(3, 1000, batch_norm)
        self.logits = nn.Linear(1000, config.num_classes)

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


class MyVGG19(_BaseModule):
    def __init__(self, config: Config):
        super(MyVGG19, self).__init__()
        in_channels = 3
        self.batch_norm = config.use_batch_norm
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
        self.fc8 = nn.Linear(4096, config.num_classes)

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


def record_train_process(config: Config, epoch: int, start_time, elapsed_time, loss: float, train_acc: float,
                         val_acc: float):
    '''
    generate temporary training process data for resuming by resume_train()
    :param self:
    :param epoch:
    :param start_time:
    :param elapsed_time:
    :param loss:
    :param train_acc:
    :param val_acc:
    :return:
    '''
    with open(config.train_record_file, 'a') as f:
        str = config.__record_dict__.format(epoch, start_time, elapsed_time, loss, train_acc, val_acc)
        f.write(str + '\n')


def resume_train(config: Config, model: nn.Module, optimizer: nn.Module = None) -> int:
    '''
    resume training process data from config.logs whice generated by record_train_process()
    :param config:
    :param model:
    :param optimizer:
    :return:
    '''
    last_epoch = -1
    if os.path.exists(config.temp_weight_path):
        try:
            model.load_state_dict(t.load(config.temp_weight_path))
        except:
            warn("Move invalid temp {} weights file {} to {}".format(type(model), config.temp_weight_path,
                                                                     config.temp_weight_path + '.badfile'))
            os.rename(config.temp_weight_path, config.temp_weight_path + '.badfile')
    if optimizer is not None and os.path.exists(config.temp_optim_path):
        try:
            optimizer.load_state_dict(t.load(config.temp_optim_path))
        except:
            warn("Move invalid temp {} weights file {} to {}".format(type(optimizer), config.temp_optim_path,
                                                                     config.temp_optim_path + '.badfile'))
            os.rename(config.temp_optim_path, config.temp_optim_path + '.badfile')
    if os.path.exists(config.train_record_file):
        try:
            with open(config.train_record_file, 'r') as f:
                last = f.readlines()[-1]
                import json
                info = json.loads(last)
                last_epoch = int(info["epoch"])
        except:
            warn("Move invalid train record file{} to {}".format(config.train_record_file,
                                                                 config.train_record_file + '.badfile'))
            warn("Can't get last_epoch value, {} will be returned".format(last_epoch))
            os.rename(config.train_record_file, config.train_record_file + '.badfile')
    return last_epoch
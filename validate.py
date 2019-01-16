# -*- coding: utf-8 -*-
# @Time    : 2019/1/9 21:19
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : validate.py
# @Software: PyCharm

import os
import math
import warnings
import numpy as np
import torch as t
from torch.nn import Module
from torchnet import meter
from collections import defaultdict
from core import Config, get_model, Visualizer, ipdb
from datasets import CloudDataLoader


def get_data(data_type, config: Config):
    data = CloudDataLoader(data_type, config)
    return data


def cover_rate2class(cover_rate: t.Tensor):
    cover_rate /= 8
    down_threshold = [-100000, 0.05, 0.1, 0.25, 0.75]
    up_threshold = [0.05, 0.1, 0.25, 0.75, 100000]
    ret = t.full([cover_rate.numel()], len(up_threshold))
    for i in range(len(up_threshold)):
        index = t.nonzero(cover_rate.gt(down_threshold[i]) * cover_rate.lt(up_threshold[i])).squeeze().cpu()
        ret.index_fill_(0, index, i)
    return ret


def validate(model, val_data, config, vis):
    # type: (Module,CloudDataLoader,Config,Visualizer)->None
    with t.no_grad():
        correct_label_distrib = defaultdict(int)  # show how many sub-images are correct for all parent-images
        error_level_distrib = defaultdict(int)  # show error levels for all parent-images
        sub_confusion_matrix = meter.ConfusionMeter(config.num_classes)
        pare_confusion_matrix = meter.ConfusionMeter(config.num_classes)
        # validate
        for i, input in enumerate(val_data):
            model.eval()
            # input data
            batch_sub_img, batch_sub_label, batch_parent_img, batch_pare_label = input
            if config.use_gpu:
                with t.cuda.device(0):
                    batch_sub_img = batch_sub_img.cuda()
                    batch_sub_label = batch_sub_label.cuda()
                    batch_parent_img = batch_parent_img.cuda()
                    batch_pare_label = batch_pare_label.cuda()
            batch_sub_label = batch_sub_label.view(-1)
            # forward
            _, batch_cover_rate, batch_prob = model(batch_sub_img, batch_parent_img)
            batch_cover_rate.squeeze_()
            # confusion matrix statistic
            batch_pred = t.argmax(batch_prob, dim=-1)
            sub_confusion_matrix.add(batch_pred, batch_sub_label)
            pare_confusion_matrix.add(cover_rate2class(batch_cover_rate), cover_rate2class(batch_pare_label))
            # distribution statistic
            equals = t.split((batch_pred == batch_sub_label), 8)
            errors = t.split((batch_pred - batch_sub_label), 8)
            for eqs, errs in zip(equals, errors):
                sub_corr_count = int(eqs.sum().cpu().numpy())
                correct_label_distrib[sub_corr_count] += 1
                sub_err_count = int(errs.abs().sum().cpu().numpy())
                error_level_distrib[sub_err_count] += 1
            for i in range(9):
                correct_label_distrib[i] += 0
            for i in range(40):
                error_level_distrib[i] += 0
            correct_label_distrib = dict(sorted(correct_label_distrib.items(), key=lambda x: x[0], reverse=False))
            error_level_distrib = dict(sorted(error_level_distrib.items(), key=lambda x: x[0], reverse=False))
            regr_dist = t.mean(t.abs(batch_cover_rate - batch_pare_label)).cpu().numpy()
            # print process
            if i % config.ckpt_freq == 0 or i >= len(val_data) - 1:
                cm_value = pare_confusion_matrix.value()
                msg = "[Validation]process:{}/{},scene regression distance:{}\n".format(i, len(val_data) - 1, regr_dist)
                msg += "confusion matrix:\n{}\ncorrect labels distribution:\n{}\nerror levels distribution:\n{}\n".format(
                    cm_value, correct_label_distrib, error_level_distrib)
                vis.log_process(i, len(val_data) - 1, msg, 'val_log', append=True)

                vis.bar(list(correct_label_distrib.values()), 'number of correct labels',
                        list(correct_label_distrib.keys()))
                vis.bar(list(error_level_distrib.values()), 'number of error levels', list(error_level_distrib.keys()))

        pare_cm = pare_confusion_matrix.value()
        sub_cm = sub_confusion_matrix.value()
        pare_acc = pare_cm.trace().astype(np.float) / pare_cm.sum()
        sub_acc = sub_cm.trace().astype(np.float) / sub_cm.sum()

    return pare_acc, pare_cm, sub_acc, sub_cm, correct_label_distrib, error_level_distrib


def main(args):
    config = Config('inference')
    print(config)
    val_data = get_data("val", config)
    model = get_model(config)
    vis = Visualizer(config)
    print("Prepare to validate model...")

    pare_acc, pare_cm, sub_acc, sub_cm, corr_label, err_level = validate(model, val_data, config, vis)
    msg = 'parent-image validation accuracy:{}\n'.format(pare_acc)
    msg += 'sub-image validation accuracy:{}\n'.format(sub_acc)
    msg += 'validation scene confusion matrix:\n{}\n'.format(pare_cm)
    msg += 'validation sub confusion matrix:\n{}\n'.format(sub_cm)
    msg += 'number of correct labels in a scene:\n{}\n'.format(corr_label)
    msg += 'number of error levels in a scene:\n{}\n'.format(err_level)
    print("Validation Finish!", msg)
    vis.log(msg, 'val_result', log_file=config.val_result)
    print("save best validation result into " + config.val_result)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)

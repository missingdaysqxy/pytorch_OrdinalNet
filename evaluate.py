# -*- coding: utf-8 -*-
# @Time    : 2019/01/10 07:53
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : evaluate.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 2019/1/9 21:19
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : validate.py
# @Software: PyCharm

import os
import warnings
import numpy as np
import torch as t
from torch.nn import Module
from torchnet import meter
from collections import defaultdict
from core import Config, get_model, OrdinalLoss, Visualizer, ipdb
from datasets import CloudDataLoader


def evaluate(model, img_paths, config):
    # type: (Module,list(str),Config)->list(list)
    with t.no_grad():
        model.eval()
        # input data
        for path in img_paths:

        if config.use_gpu:
            with t.cuda.device(0):
                batch_sub_img = batch_sub_img.cuda()
                batch_sub_label = batch_sub_label.cuda()
                batch_parent_img = batch_parent_img.cuda()
                batch_pare_label = batch_pare_label.cuda()
        batch_sub_label = batch_sub_label.view(-1)
        # forward
        batch_inter_prob, batch_cover_rate, batch_final_prob = model(batch_sub_img, batch_parent_img)
        batch_cover_rate.squeeze_()
        # confusion matrix statistic
        batch_inter_pred = t.argmax(batch_inter_prob, dim=-1)
        batch_final_pred = t.argmax(batch_final_prob, dim=-1)
        inter_confusion_matrix.add(batch_inter_pred, batch_sub_label)
        final_confusion_matrix.add(batch_final_pred, batch_sub_label)
        # distribution statistic
        compares = (batch_final_pred == batch_sub_label)
        compares = t.split(compares, 8)
        for cps in compares:
            sub_corr_count = cps.sum().cpu().numpy()
            num_correct_distribute[int(sub_corr_count)] += 1
        regr_dist = t.mean(t.abs(batch_cover_rate - batch_pare_label)).cpu().numpy()
        # print process
        if i % config.ckpt_freq == 0 or i >= len(val_data) - 1:
            msg = "[Validation]process:{}/{},scene regression regr_dist:{},inter confusion matrix:\n{}\nconfusion matrix:\n{}".format(
                i, len(val_data) - 1, regr_dist, inter_confusion_matrix.value(), final_confusion_matrix.value())
            vis.log_process(i, len(val_data) - 1, msg, 'val_log', append=True)
        # del batch_img, batch_label, batch_scene, batch_probs, batch_final_pred, compares
        # t.cuda.empty_cache()
    inter_cm = inter_confusion_matrix.value()
    final_cm = final_confusion_matrix.value()
    inter_accuracy = inter_cm.trace().astype(np.float) / inter_cm.sum()
    final_accuracy = final_cm.trace().astype(np.float) / final_cm.sum()
    return inter_accuracy, inter_confusion_matrix.value(), final_accuracy, final_confusion_matrix.value(), num_correct_distribute


def main(args):
    config = Config('inference')
    print(config)
    model = get_model(config)
    if isinstance(args.input,str):
        img_path=[args.input]
    elif isinstance(args.input,list):
        img_path=[]
        for path in args.input:

        print("Prepare to evaluate %s" % img_path)
        inter_acc, inter_cm, val_acc, val_cm, num_dis = validate(model, val_data, config, vis)
        msg = 'interm accuracy:{}, validation accuracy:{}\n'.format(inter_acc, val_acc)
        msg += 'interm confusion matrix:\n{}\nvalidation confusion matrix:\n{}\n'.format(inter_cm, val_cm)
        msg += 'number of correct labels in a scene:\n{}'.format(num_dis)
        print("Validation Finish!", msg)
        vis.log(msg, 'val_result', log_file='val_result.txt')


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('input', type=str, help='Scene Image(s) to evaluate.')
    parse.add_argument('-o', '--output', type=str, help='write evaluation result into this file')
    args = parse.parse_args()

    main(args)

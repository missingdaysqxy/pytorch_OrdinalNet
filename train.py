# -*- coding: utf-8 -*-
# @Time    : 2019/1/5 18:05
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : train.py
# @Software: PyCharm

import os
import time
import numpy as np
import torch as t
from warnings import warn
from torch.nn import Module
from torchnet import meter
from core import *
from validate import validate as val
from .datasets import CloudDataLoader


def get_data(data_type, config: Config):
    data = CloudDataLoader(data_type, config)
    return data


def get_loss_functions(config: Config) -> Module:
    if config.loss_type == "ordinal":
        return t.nn.CrossEntropyLoss(), t.nn.MSELoss(), OrdinalLoss()
    elif config.loss_type in ["cross_entropy", "crossentropy", "cross", "ce"]:
        return t.nn.CrossEntropyLoss(), t.nn.MSELoss(), t.nn.CrossEntropyLoss
    else:
        raise RuntimeError("Invalid config.loss_type:" + config.loss_type)


def get_optimizer(model: Module, config: Config) -> t.optim.Optimizer:
    if config.optimizer == "sgd":
        return t.optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        return t.optim.Adam(model.parameters())
    else:
        raise RuntimeError("Invalid value: config.optimizer")


def train(model, train_data, val_data, config, vis):
    # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visualizer)->None
    # init loss and optim
    criterion1, criterion2, criterion3 = get_loss_functions(config)
    optimizer = get_optimizer(model, config)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay)
    # try to resume
    last_epoch = resume_checkpoint(optimizer, config)
    assert last_epoch + 1 < config.max_epoch, \
        "previous training has reached epoch {}, please increase the max_epoch in {}".format(last_epoch + 1,
                                                                                             type(config))
    # init meter statistics
    loss_meter = meter.AverageValueMeter()
    loss1_meter = meter.AverageValueMeter()
    loss2_meter = meter.AverageValueMeter()
    loss3_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(config.num_classes)
    for epoch in range(last_epoch + 1, config.max_epoch):
        epoch_start = time.time()
        loss_mean = None
        train_acc = 0
        scheduler.step(epoch)
        loss_meter.reset()
        confusion_matrix.reset()
        model.train()
        for i, input in enumerate(train_data):
            # input data
            batch_subs, batch_label_s, batch_parent, batch_label_p = input
            if config.use_gpu:
                batch_subs = batch_subs.cuda()
                batch_label_s = batch_label_s.cuda()
                batch_parent = batch_parent.cuda()
                batch_label_p = batch_label_p.cuda()
                criterion1 = criterion1.cuda()
                criterion2 = criterion2.cuda()
                criterion3 = criterion3.cuda()
            batch_parent.requires_grad_(True)
            batch_subs.requires_grad_(True)
            batch_label_p.requires_grad_(False)
            batch_label_s.requires_grad_(False)
            # forward
            probs_c, regrs, probs_final = model(batch_parent, batch_subs)
            loss1 = criterion1(probs_c, batch_label_s)
            loss2 = criterion2(regrs, batch_label_p)
            loss3 = criterion3(probs_final, batch_label_s)
            loss = loss1 + loss2 + loss3
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistic
            loss_meter.add(loss.data.cpu())
            loss1_meter.add(loss1.data.cpu())
            loss2_meter.add(loss2.data.cpu())
            loss3_meter.add(loss3.data.cpu())
            confusion_matrix.add(probs_final.data, batch_label_s.data)
            # print process
            if i % config.ckpt_freq == config.ckpt_freq - 1:
                step = epoch * len(train_data) + i
                loss_mean = loss_meter.value()[0]
                loss1_mean = loss1_meter.value()[0]
                loss2_mean = loss2_meter.value()[0]
                loss3_mean = loss3_meter.value()[0]
                cm_value = confusion_matrix.value()
                num_correct = cm_value.trace().astype(np.float)
                train_acc = 100 * num_correct / cm_value.sum()
                vis.plot(loss_mean, step, 'loss:' + config.loss_type)
                vis.plot(loss1_mean, step, 'loss1:' + config.loss_type)
                vis.plot(loss2_mean, step, 'loss2:' + config.loss_type)
                vis.plot(loss3_mean, step, 'loss3:' + config.loss_type)
                vis.plot(train_acc, step, 'train_accuracy')
                lr = optimizer.param_groups[0]['lr']
                msg = "epoch:{},iteration:{}/{},loss:{},train_accuracy:{},lr:{},confusion_matrix:\n{}".format(
                    epoch, i, len(train_data), loss_mean, train_acc, lr, confusion_matrix.value())
                vis.log_process(i, len(train_data), msg, 'train_log')

                # check if debug file occur
                if os.path.exists(config.debug_flag_file):
                    ipdb.set_trace()
        # validate after each epoch
        val_acc, confusion_matrix = val(model, val_data, config, vis)
        vis.plot(val_acc, epoch, 'val_accuracy')
        # save checkpoint
        make_checkpoint(config, epoch, epoch_start, loss_mean, train_acc, val_acc, model, optimizer)


def main(args):
    config = Config('train')
    print(config)
    train_data = get_data("train", config)
    val_data = get_data("val", config)
    model = get_model(config)
    vis = Visualizer(config)
    print("Prepare to train core...")
    train(model, train_data, val_data, config, vis)
    # save core
    print("Training Finish! Saving core...")
    t.save(model.state_dict(), config.weight_save_path)
    os.remove(config.temp_optim_path)
    os.remove(config.temp_weight_path)
    print("Model saved into " + config.weight_save_path)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)

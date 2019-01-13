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


def get_data(data_type, config: Config):
    data = CloudDataLoader(data_type, config)
    return data


def get_loss_function(config: Config) -> Module:
    if config.loss_type == "ordinal":
        return OrdinalLoss()
    elif config.loss_type in ["cross_entropy", "crossentropy", "cross", "ce"]:
        return t.nn.CrossEntropyLoss()
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
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay)
    # try to resume
    last_epoch = resume_checkpoint(optimizer, config)
    assert last_epoch + 1 < config.max_epoch, \
        "previous training has reached epoch {}, please increase the max_epoch in {}".format(last_epoch + 1,
                                                                                             type(config))
    # init meter statistics
    loss_meter = meter.AverageValueMeter()
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
            batch_img, batch_label, batch_scene = input
            batch_img = V(batch_img)
            batch_label = V(batch_label)
            if config.use_gpu:
                batch_img = batch_img.cuda()
                batch_label = batch_label.cuda()
                criterion = criterion.cuda()
            # forward
            batch_probs = model(batch_img)
            loss = criterion(batch_probs, batch_label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistic
            loss_meter.add(loss.data.cpu())
            confusion_matrix.add(batch_probs.data, batch_label.data)
            # print process
            if i % config.ckpt_freq == config.ckpt_freq - 1:
                step = epoch * len(train_data) + i
                loss_mean = loss_meter.value()[0]
                cm_value = confusion_matrix.value()
                num_correct = cm_value.trace().astype(np.float)
                train_acc = 100 * num_correct / cm_value.sum()
                vis.plot(loss_mean, step, 'loss:' + config.loss_type)
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

        record_train_process(config, epoch, epoch_start, time.time() - epoch_start, loss_mean, train_acc, val_acc)


# def val(core, val_data, config, visdom):
#     # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visdom)->[float,meter.ConfusionMeter]
#     core.eval()
#     confusion_matrix = meter.ConfusionMeter(config.num_classes)
#     for i, input in enumerate(val_data):
#         # input data
#         batch_img, batch_label, batch_scene = input
#         batch_img = V(batch_img)
#         batch_label = V(batch_label)
#         if config.use_gpu:
#             batch_img = batch_img.cuda()
#             batch_label = batch_label.cuda()
#         # forward
#         batch_probs = core(batch_img)
#         confusion_matrix.add(batch_probs.data.squeeze(), batch_label.data.long())
#         if i % config.ckpt_freq == config.ckpt_freq - 1:
#             msg = "[Validation]process:{}/{}, confusion matrix:\n{}".format(
#                 i, len(val_data), confusion_matrix.value())
#             log_process(i, len(val_data), msg, visdom, 'val_log', append=True)
#     cm_value = confusion_matrix.value()
#     num_correct = cm_value.trace().astype(np.float)
#     accuracy = 100 * num_correct / cm_value.sum()
#     return accuracy, confusion_matrix


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

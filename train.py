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
from collections import defaultdict
from torch.nn import Module
from torchnet import meter
from core import *
from validate import validate as val
from datasets import CloudDataLoader

LOSS_WEIGHT = [2, 0.4, 100]


def get_data(data_type, config: Config):
    data = CloudDataLoader(data_type, config)
    return data


def get_loss_functions(config: Config) -> Module:
    if config.loss_type == "ordinal":
        return t.nn.CrossEntropyLoss(), t.nn.MSELoss(), OrdinalLoss()
    elif config.loss_type in ["cross_entropy", "crossentropy", "cross", "ce"]:
        return t.nn.CrossEntropyLoss(), t.nn.MSELoss(), t.nn.CrossEntropyLoss()
    else:
        raise RuntimeError("Invalid config.loss_type:" + config.loss_type)


def get_optimizer(model: Module, config: Config) -> t.optim.Optimizer:
    if config.optimizer == "sgd":
        return t.optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        return t.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError("Invalid value: config.optimizer")


def train(model, train_data, val_data, config, vis):
    # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visualizer)->None
    def loss_sum(c1, c2, c3):
        c1 *= LOSS_WEIGHT[0]
        c2 *= LOSS_WEIGHT[1]
        c3 *= LOSS_WEIGHT[2]
        sum = c1 + c2 + c3
        loss1 = c1 * (c2 + c3) / sum
        loss2 = c2 * (c1 + c3) / sum
        loss3 = c3 * (c1 + c2) / sum
        loss = loss1 + loss2 + loss3
        return loss, loss1, loss2, loss3

    # init loss and optim
    criterion1, criterion2, criterion3 = get_loss_functions(config)
    optimizer = get_optimizer(model, config)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay)
    # try to resume
    last_epoch = resume_checkpoint(config, model, optimizer)
    assert last_epoch + 1 < config.max_epoch, \
        "previous training has reached epoch {}, please increase the max_epoch in {}". \
            format(last_epoch + 1, type(config))
    if last_epoch == -1:  # start a new train proc
        vis.save(config.vis_env_path + 'last')
        vis.clear()
    # init meter statistics
    loss_meter = meter.AverageValueMeter()
    loss1_meter = meter.AverageValueMeter()
    loss2_meter = meter.AverageValueMeter()
    loss3_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(config.num_classes)
    last_accuracy = 0
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
            batch_sub_img, batch_sub_label, batch_parent_img, batch_pare_label = input
            if config.use_gpu:
                with t.cuda.device(0):
                    batch_sub_img = batch_sub_img.cuda()
                    batch_sub_label = batch_sub_label.cuda()
                    batch_parent_img = batch_parent_img.cuda()
                    batch_pare_label = batch_pare_label.cuda()
                    criterion1 = criterion1.cuda()
                    criterion2 = criterion2.cuda()
                    criterion3 = criterion3.cuda()
            batch_sub_img.requires_grad_(True)
            batch_parent_img.requires_grad_(True)
            batch_sub_label.requires_grad_(False)
            batch_pare_label.requires_grad_(False)
            batch_sub_label = batch_sub_label.view(-1)
            # forward
            batch_interim_prob, batch_cover_rate, batch_final_prob = model(batch_sub_img, batch_parent_img)
            c1 = criterion1(batch_interim_prob, batch_sub_label)
            c2 = criterion2(batch_cover_rate, batch_pare_label)
            c3 = criterion3(batch_final_prob, batch_sub_label)
            loss, loss1, loss2, loss3 = loss_sum(c1, c2, c3)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistic
            loss_meter.add(loss.data.cpu())
            loss1_meter.add(loss1.data.cpu())
            loss2_meter.add(loss2.data.cpu())
            loss3_meter.add(loss3.data.cpu())
            batch_final_pred = t.argmax(batch_final_prob, dim=-1)
            confusion_matrix.add(batch_final_pred, batch_sub_label)
            # print process
            if i % config.ckpt_freq == 0 or i >= len(train_data) - 1:
                step = epoch * len(train_data) + i
                loss_mean = loss_meter.value()[0]
                loss1_mean = loss1_meter.value()[0]
                loss2_mean = loss2_meter.value()[0]
                loss3_mean = loss3_meter.value()[0]
                cm_value = confusion_matrix.value()
                num_correct = cm_value.trace().astype(np.float)
                train_acc = num_correct / cm_value.sum()
                vis.plot(loss_mean, step, 'Loss_Sum', "Loss Curve", ["Loss_Sum", "Loss1", "Loss2", "Loss3"])
                vis.plot(loss1_mean, step, 'Loss1', "Loss Curve")
                vis.plot(loss2_mean, step, 'Loss2', "Loss Curve")
                vis.plot(loss3_mean, step, 'Loss3', "Loss Curve")
                vis.plot(train_acc, step, 'train_acc', 'Training Accuracy')
                lr = optimizer.param_groups[0]['lr']
                msg = "epoch:{},iteration:{}/{},loss:{},loss1:{},loss2:{},loss3:{},train_accuracy:{},lr:{},confusion_matrix:\n{}".format(
                    epoch, i, len(train_data) - 1, loss_mean, loss1_mean, loss2_mean, loss3_mean,
                    train_acc, lr, confusion_matrix.value())
                vis.log_process(i, len(train_data) - 1, msg, 'train_log')

                # check if debug file occur
                if os.path.exists(config.debug_flag_file):
                    ipdb.set_trace()
        # validate after each epoch
        pare_acc, pare_cm, sub_acc, sub_cm, corr_label, err_level = val(model, val_data, config, vis)
        vis.plot(pare_acc, epoch, 'pare_acc', 'Parent-Images Validation Accuracy', ['pare_acc'])
        vis.plot(sub_acc, epoch, 'sub_acc', 'Sub-Images Validation Accuracy', ['sub_acc'])
        # save checkpoint
        if pare_acc > last_accuracy:
            msg += 'best validation result after epoch {}, loss:{}, train_acc: {}'.format(epoch, loss_mean, train_acc)
            msg += 'parent-image validation accuracy:{}\n'.format(pare_acc)
            msg += 'sub-image validation accuracy:{}\n'.format(sub_acc)
            msg += 'validation scene confusion matrix:\n{}\n'.format(pare_cm)
            msg += 'validation sub confusion matrix:\n{}\n'.format(sub_cm)
            msg += 'number of correct labels in a scene:\n{}\n'.format(corr_label)
            msg += 'number of error levels in a scene:\n{}\n'.format(err_level)
            vis.log(msg, 'best_val_result', log_file=config.val_result, append=False)
            print("save best validation result into " + config.val_result)
        last_accuracy = pare_acc
        make_checkpoint(config, epoch, epoch_start, loss_mean, train_acc, sub_acc, model, optimizer)


def main(*args, **kwargs):
    config = Config('train', **kwargs)
    print(config)
    train_data = get_data("train", config)
    val_data = get_data("val", config)
    model = get_model(config)
    vis = Visualizer(config)
    print("Prepare to train model...")
    train(model, train_data, val_data, config, vis)
    # save core
    print("Training Finish! Saving model...")
    try:
        t.save(model.state_dict(), config.weight_save_path)
        os.remove(config.temp_optim_path)
        os.remove(config.temp_weight_path)
        print("Model saved into " + config.weight_save_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to save model because {}, check temp weight file in {}".format(e, config.temp_weight_path))


if __name__ == '__main__':
    # config = Config('train')
    #
    # model = get_model(config)
    #
    # print(model)

    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)

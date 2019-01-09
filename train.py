# -*- coding: utf-8 -*-
# @Time    : 2019/1/5/005 18:05 下午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : train.py
# @Software: PyCharm

import os
import warnings
import numpy as np
import torch as t
from torch.autograd import Variable as V
from torch.nn import Module
from visdom import Visdom
from torchnet import meter
from model import CloudDataLoader, Config, OrdinalNet, OrdinalLoss, plot, log_process, ipdb


def get_model(config: Config):
    weight_path = config.weight_load_path
    if os.path.exists(weight_path):
        pretrained = False
    else:
        pretrained = config.load_public_weight
    model = OrdinalNet(config.num_classes, config.use_batch_norm, pretrained)
    if os.path.exists(weight_path):
        try:
            model.load_state_dict(t.load(weight_path))
            print('loaded weight from ' + weight_path)
        except:
            warnings.warn('Failed to load weight file ' + weight_path)
            model.initialize_weights()
    return model
    # return vgg.vgg19_bn(pretrained, num_classes=config.num_classes)


def get_data(data_path, config: Config, shuffle):
    data = CloudDataLoader(data_path, config.classes_list, config.batch_size, config.image_resize, shuffle,
                           config.num_data_workers)
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


def resume_train(optimizer: Module, config: Config):
    lastepoch = -1
    if os.path.exists(config.temp_optim_path):
        try:
            optimizer.load_state_dict(t.load(config.temp_optim_path))
        except:
            warnings.warn("Invalid temp optimizer file " + config.temp_optim_path)
            os.rename(config.temp_optim_path, config.temp_optim_path + '.badfile')
    if os.path.exists(config.result_file):
        try:
            with open(config.result_file, 'r') as f:
                last = f.readlines()[-1]
                lastepoch = int(last)
        except:
            warnings.warn("Can't read lastepoch from file " + config.result_file)
    return lastepoch, optimizer


def train(model, train_data, val_data, config, visdom):
    # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visdom)->None
    os.makedirs(os.path.dirname(config.temp_weight_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.temp_optim_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.result_file), exist_ok=True)
    # move model to GPU
    if config.use_gpu:
        model = model.cuda()
        model = t.nn.DataParallel(model, config.gpu_list)
    # init loss and optim
    criterion = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, 1, config.lr_decay)
    # try to resume
    lastepoch, optimizer = resume_train(optimizer, config)
    # init meter statistics
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(config.num_classes)
    with open(config.result_file, 'a') as file:
        for epoch in range(lastepoch + 1, config.max_epoch):
            scheduler.step(epoch)
            loss_meter.reset()
            confusion_matrix.reset()
            model.train()
            for i, input in enumerate(train_data):
                # input data
                batch_img, batch_label, batch_parent = input
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
                    accuracy = 100 * num_correct / cm_value.sum()
                    plot(loss_mean, step, visdom, 'loss:' + config.loss_type)
                    plot(accuracy, step, visdom, 'train_accuracy')
                    lr = optimizer.param_groups[0]['lr']
                    msg = "epoch:{},iteration:{}/{},loss:{},train_accuracy:{},lr:{},confusion_matrix:\n{}".format(
                        epoch, i, len(train_data), loss_mean, accuracy, lr, confusion_matrix.value())
                    log_process(i, len(train_data), msg, visdom, 'train_log')
                    # save checkpoint
                    t.save(model.state_dict(), config.temp_weight_path)
                    t.save(optimizer.state_dict(), config.temp_optim_path)
                    # check if debug file occur
                    if os.path.exists(config.debug_flag_file):
                        ipdb.set_trace()
            # validation after each epoch
            accuracy, confusion_matrix = val(model, val_data, config, visdom)
            plot(accuracy, epoch, visdom, 'val_accuracy')
            # log epoch num
            file.write(str(epoch) + '\n')


def val(model, val_data, config, visdom):
    # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visdom)->[float,meter.ConfusionMeter]
    model.eval()
    confusion_matrix = meter.ConfusionMeter(config.num_classes)
    for i, input in enumerate(val_data):
        # input data
        batch_img, batch_label, batch_parent = input
        batch_img = V(batch_img)
        batch_label = V(batch_label)
        if config.use_gpu:
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
        # forward
        batch_probs = model(batch_img)
        confusion_matrix.add(batch_probs.data.squeeze(), batch_label.data.long())
        if i % config.ckpt_freq == config.ckpt_freq - 1:
            msg = "[Validation]process:{}/{}, confusion matrix:\n{}".format(
                i, len(val_data), confusion_matrix.value())
            log_process(i, len(val_data), msg, visdom, 'val_log', append=True)
    cm_value = confusion_matrix.value()
    num_correct = cm_value.trace().astype(np.float)
    accuracy = 100 * num_correct / cm_value.sum()
    return accuracy, confusion_matrix


def main(args):
    config = Config()
    print(config)
    train_data = get_data(config.train_data_root, config, shuffle=config.shuffle_train)
    val_data = get_data(config.val_data_root, config, shuffle=config.shuffle_val)
    model = get_model(config)
    try:
        vis = Visdom(env=config.visdom_env)
        if not vis.check_connection():
            raise ConnectionError("Can't connect to visdom server, please run command 'python -m visdom.server'")
    except:
        warnings.warn("Can't open Visdom!")
    print("Prepare to train model...")
    train(model, train_data, val_data, config, vis)
    # save model
    print("Training Finish! Saving model...")
    t.save(model.state_dict(), config.weight_save_path)
    os.remove(config.temp_optim_path)
    os.remove(config.temp_weight_path)
    print("Model saved into " + config.weight_save_path)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)

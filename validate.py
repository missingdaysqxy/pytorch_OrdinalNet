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
from torch.autograd import Variable as V
from torch.nn import Module
from visdom import Visdom
from torchnet import meter
from collections import defaultdict
from model import CloudDataLoader, Config, OrdinalNet, OrdinalLoss, plot, log, log_process, ipdb


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


def validate(model, val_data, config, visdom):
    # type: (Module,CloudDataLoader,CloudDataLoader,Config,Visdom)->None
    # move model to GPU
    if config.use_gpu:
        model = model.cuda()
        model = t.nn.DataParallel(model, config.gpu_list)
    # validate
    scene_dict = defaultdict(list)
    confusion_matrix = meter.ConfusionMeter(config.num_classes)
    for i, (batch_img, batch_label, batch_scene) in enumerate(val_data):
        model.eval()
        # input data
        batch_img = V(batch_img)
        batch_label = V(batch_label)
        if config.use_gpu:
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
        # forward
        batch_probs = model(batch_img)
        preds = t.argmax(batch_probs, dim=-1)
        compares = (preds == batch_label)
        if compares.is_cuda:
            compares = compares.cpu()
        for scene, c in zip(batch_scene, compares.numpy()):
            scene_dict[scene].append(int(c))
        confusion_matrix.add(batch_probs.data.squeeze(), batch_label.data.long())
        if i % config.ckpt_freq == 0:
            msg = "[Validation]process:{}/{}, confusion matrix:\n{}".format(
                i, len(val_data), confusion_matrix.value())
            log_process(i, len(val_data), msg, visdom, 'val_log', append=True)
        del batch_img, batch_label, batch_scene, batch_probs, preds, compares
        t.cuda.empty_cache()

    cm_value = confusion_matrix.value()
    num_correct = cm_value.trace().astype(np.float)
    accuracy = 100 * num_correct / cm_value.sum()
    # summaries
    scene_sum = defaultdict(int)  # summary of correct scenes distribution
    for k, v in scene_dict.items():
        account = np.sum(v)
        scene_sum[account] += 1
    return accuracy, confusion_matrix, scene_sum


def main(args):
    config = Config()
    print(config)
    val_data = get_data(config.val_data_root, config, shuffle=config.shuffle_val)
    model = get_model(config)
    try:
        vis = Visdom(env=config.visdom_env)
        if not vis.check_connection():
            raise ConnectionError("Can't connect to visdom server, please run command 'python -m visdom.server'")
    except:
        warnings.warn("Can't open Visdom!")
    print("Prepare to validate model...")
    accuracy, confusion_matrix, scene_sum = validate(model, val_data, config, vis)
    # plot(accuracy, 0, visdom, 'test_accuracy')
    msg = 'test_accuracy:{}\naccuracy summaries:{}\nconfusion matrix:\n{}'. \
        format(accuracy, scene_sum, confusion_matrix)
    log(msg, vis, 'test_result', logfile='test_result.txt')


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()

    main(args)

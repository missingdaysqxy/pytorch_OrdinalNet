# -*- coding: utf-8 -*-
# @Time    : 2019/1/6/006 4:23 上午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : config.py
# @Software: PyCharm

import os
from time import strftime as timestr


class Config(object):
    train_data_root = r'I:\ordinal_clouds\separate_relabel\train'
    val_data_root = r'I:\ordinal_clouds\separate_relabel\validation'
    shuffle_train = True
    shuffle_val = True

    load_public_weight = True
    weight_load_path = r'checkpoints/crossentropy_vgg.pth'
    weight_save_path = r'checkpoints/ordinal_vgg.pth'
    log_root = r'logs'
    debug_flag_file = r'debug'
    classes_list = ['A', 'B', 'C', 'D', 'E', 'nodata']

    gpu_list = [0]
    num_data_workers = 4
    visdom_env = 'main'

    max_epoch = 10
    batch_size = 16
    ckpt_freq = 10  # save checkpoint after these iterations

    loss_type = "ordinal"
    optimizer = "adam"
    lr = 0.01  # learning rate
    lr_decay = 0.95
    momentum = 0.9
    weight_decay = 1e-4  # weight decay (L2 penalty)

    use_batch_norm = True

    image_resize = [224, 224]

    def __init__(self):
        self.starttime = timestr('%m%d%H%M%S')
        from torch.cuda import is_available as cuda_availiable
        self.loss_type = self.loss_type.lower()
        assert self.loss_type in ["ordinal", "cross_entropy", "crossentropy", "cross", "ce"]
        self.optimizer = self.optimizer.lower()
        assert self.optimizer in ["sgd", "adam"]
        self.num_gpu = len(self.gpu_list)
        self.use_gpu = cuda_availiable() and self.num_gpu > 0
        self.num_classes = len(self.classes_list)
        assert self.batch_size % self.num_gpu == 0, "Can't split a batch of {} datas averagely into {} gpus".format(
            self.batch_size, self.num_gpu)
        self.temp_weight_path = os.path.join(self.log_root, 'tmpmodel{}.pth'.format(self.starttime))
        self.temp_optim_path = os.path.join(self.log_root, 'tmp{}{}.pth'.format(self.optimizer, self.starttime))
        self.result_file = os.path.join(self.log_root, 'result.csv')

    def __str__(self):
        """:return Configuration values."""
        str = "Configurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                str += "{:30} {}\n".format(a, getattr(self, a))
        return str


def main(args):
    config = Config()
    print(config)


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    args = parse.parse_args()
    main(args)

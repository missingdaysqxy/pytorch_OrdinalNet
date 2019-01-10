# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 4:23
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : config.py
# @Software: PyCharm

import os
import warnings
from time import strftime as timestr


class Config(object):
    # data config
    train_data_root = r'I:\ordinal_clouds\separate_relabel\train'
    val_data_root = r'I:\ordinal_clouds\separate_relabel\validation'
    classes_list = ['A', 'B', 'C', 'D', 'E', 'nodata']
    shuffle_train = True
    shuffle_val = True
    drop_last_train = True
    drop_last_val = False

    # weight S/L config
    load_public_weight = True
    weight_load_path = r'ordinal_vgg.pth'
    weight_save_path = r'checkpoints/ordinal_vgg.pth'
    log_root = r'logs'

    debug_flag_file = r'debug'

    # efficiency config
    gpu_list = [0]
    num_data_workers = 4
    max_epoch = 15
    batch_size = 16

    # module config
    # module_name = "OrdinalNet"
    image_resize = [224, 224]
    use_batch_norm = True
    loss_type = "ce"
    optimizer = "adam"
    lr = 0.01  # learning rate
    lr_decay = 0.95
    momentum = 0.9
    weight_decay = 1e-4  # weight decay (L2 penalty)

    # visualize config
    visdom_env = 'main'
    ckpt_freq = 10  # save checkpoint after these iterations

    def __init__(self):
        self.starttime = timestr('%Y%m%d%H%M%S')
        self.loss_type = self.loss_type.lower()
        assert self.loss_type in ["ordinal", "cross_entropy", "crossentropy", "cross", "ce"]
        self.optimizer = self.optimizer.lower()
        assert self.optimizer in ["sgd", "adam"]
        if isinstance(self.gpu_list, list):
            pass
        elif self.gpu_list is None:
            self.gpu_list = []
        elif isinstance(self.gpu_list, int):
            self.gpu_list = [self.gpu_list]
        elif isinstance(self.gpu_list, str):
            self.gpu_list = [int(x) for x in self.gpu_list.split(',')]
        else:
            raise AttributeError(
                "Invalid gpu_list, expect a number list. See `device_ids` of https://pytorch.org/docs/stable/nn.html#dataparallel for more details")
        self.num_gpu = len(self.gpu_list)
        from torch.cuda import is_available as cuda_availiable
        self.use_gpu = cuda_availiable() and self.num_gpu > 0
        self.num_classes = len(self.classes_list)
        assert self.batch_size % self.num_gpu == 0, "Can't split a batch of {} datas averagely into {} gpus".format(
            self.batch_size, self.num_gpu)
        os.makedirs(self.log_root, exist_ok=True)
        self.temp_weight_path = os.path.join(self.log_root, 'tmpmodel{}.pth'.format(self.starttime))
        self.temp_optim_path = os.path.join(self.log_root, 'tmp{}{}.pth'.format(self.optimizer, self.starttime))
        self.log_path = os.path.join(self.log_root, 'logfile{}.txt'.format(self.starttime))
        self.train_record_file = os.path.join(self.log_root, 'train_record.jsons.txt')
        # reocrd training process with informations of [epoch, start time, elapsed time, loss value, train accuracy, validate accuracy]
        # DO NOT CHANGE IT!!!
        self.__record_fields__ = ['epoch', 'start', 'elapsed', 'loss', 'train_acc', 'val_acc']
        # 'val_cm', '']  # validate confusion matrix,
        if len(self.__record_fields__) == 0:
            warnings.warn(
                '__record_fields__ of {} is empty, this may cause unknown issues when recording training informations into {}'.format(
                    type(self), self.train_record_file))
            self.__record_dict__ = '{{}}'
        else:
            self.__record_dict__ = '{{'
            for field in self.__record_fields__:
                self.__record_dict__ += '"{}":"{{}}",'.format(field)
            self.__record_dict__ = self.__record_dict__[:-1] + '}}'

    def __str__(self):
        """:return Configuration details."""
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

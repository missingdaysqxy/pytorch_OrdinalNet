# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 4:23
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : config.py
# @Software: PyCharm

import os
from warnings import warn
from time import strftime as timestr


class Config(object):
    # data config
    train_data_root = r'D:\qixuan\Documents\PythonProjects\clouds_data\mode_2004\train'
    val_data_root = r'D:\qixuan\Documents\PythonProjects\clouds_data\mode_2004\validation'
    classes_list = ['A', 'B', 'C', 'D', 'E', 'nodata']
    shuffle_train = True
    shuffle_val = True
    drop_last_train = True
    drop_last_val = False

    # efficiency config
    use_gpu = True      # if there's no cuda-available GPUs, this will turn to False automatically
    num_data_workers = 4    # how many subprocesses to use for data loading
    pin_memory = True    # only set to True when your machine's memory is large enough
    max_epoch = 15
    batch_size = 16

    # weight S/L config
    load_public_weight = True
    weight_load_path = r'checkpoints/crossentropy_vgg.pth'
    weight_save_path = r'checkpoints/crossentropy_vgg.pth'
    log_root = r'logs'

    debug_flag_file = r'debug'

    # module config
    module_name = "OrdinalNet"
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

    def __init__(self, mode):
        assert mode in ['train', 'inference']
        self.grad_enable = mode == 'train'
        self.start_time = timestr('%Y%m%d%H%M%S')
        # data config
        self.num_classes = len(self.classes_list)
        # efficiency config
        if self.use_gpu:
            from torch.cuda import is_available as cuda_available, device_count
            if cuda_available():
                self.num_gpu = device_count()
                self.gpu_list = list(range(self.num_gpu))
                assert self.batch_size % self.num_gpu == 0, \
                    "Can't split a batch of {} datas averagely into {} gpus".format(self.batch_size, self.num_gpu)
            else:
                warn("There is no available cuda devices on this machine, use_gpu will be set to False.")
                self.use_gpu = False
                self.num_gpu = 0
                self.gpu_list = []
        else:
            from torch.cuda import is_available as cuda_available
            if cuda_available():
                warn("Available cuda devices were found, please switch use_gpu to True for acceleration.")
            self.num_gpu = 0
            self.gpu_list = []
        if self.use_gpu:
            self.map_location = lambda storage, loc: storage
        else:
            self.map_location = "cpu"
        # weight S/L config
        if self.load_public_weight and not os.path.exists(self.weight_load_path):
            warn("Can't find weight file in config.weight_load_path: "
                 + self.weight_load_path) + "\nPretrained weight by PyTorch will be used"
            self.use_pytorch_weight = True
        else:
            self.use_pytorch_weight = False
        os.makedirs(self.log_root, exist_ok=True)
        assert os.path.isdir(self.log_path)
        self.vis_env_path = os.path.join(self.log_path, 'visdom_{}.json'.format(self.visdom_env))
        self.temp_weight_path = os.path.join(self.log_root, 'tmpmodel{}.pth'.format(self.start_time))
        self.temp_optim_path = os.path.join(self.log_root, 'tmp{}{}.pth'.format(self.optimizer, self.start_time))
        self.log_path = os.path.join(self.log_root, 'logfile{}.txt'.format(self.start_time))
        self.train_record_file = os.path.join(self.log_root, 'train_record.jsons.txt')
        """
       reocrd training process with informations of 
       [epoch, start time, elapsed time, loss value, train accuracy, validate accuracy]
       DO NOT CHANGE IT unless you know what you're doing!!!
       """
        self.__record_fields__ = ['epoch', 'start', 'elapsed', 'loss', 'train_acc', 'val_acc']
        if len(self.__record_fields__) == 0:
            warn(
                '__record_fields__ of {} is empty, this may cause unknown issues when recording training informations into {}'.format(
                    type(self), self.train_record_file))
            self.__record_dict__ = '{{}}'
        else:
            self.__record_dict__ = '{{'
            for field in self.__record_fields__:
                self.__record_dict__ += '"{}":"{{}}",'.format(field)
            self.__record_dict__ = self.__record_dict__[:-1] + '}}'
        # module config
        self.loss_type = self.loss_type.lower()
        assert self.loss_type in ["ordinal", "cross_entropy", "crossentropy", "cross", "ce"]
        self.optimizer = self.optimizer.lower()
        assert self.optimizer in ["sgd", "adam"]

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

# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 5:48
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : __init__.py
# @Software: PyCharm

from .models import _BaseModule, OrdinalNet, MyVGG19
from .datasets import CloudDataLoader, CloudDataset, _isArrayLike
from .config import Config
from .utils import Visualizer
from .ordinalloss import OrdinalLoss

try:
    import ipdb
except:
    import pdb as ipdb


def get_model(config: Config, **kwargs) -> _BaseModule:
    assert isinstance(config, Config)
    from os.path import join, exists
    from torch import load, nn, set_grad_enabled
    try:
        with set_grad_enabled(config.grad_enable):
            model = getattr(models, config.module_name)(config, **kwargs)
    except AttributeError as e:
        import sys
        raise AttributeError(
            "No module named '{}' exists in {}".format(config.module_name, join(sys.path[0], "models.py")))
    from warnings import warn
    # move model to GPU
    if config.use_gpu:
        model = model.cuda()
    # initialize weights
    try:
        getattr(model, "initialize_weights")()
    except AttributeError as e:
        warn("try to initialize weights failed because:\n" + str(e))
    # parallel processing
    model = nn.DataParallel(model, config.gpu_list)
    # load weights
    if exists(config.weight_load_path):
        try:
            model.load_state_dict(load(config.weight_load_path, map_location=config.map_location))
            print('Loaded weights from ' + config.weight_load_path)
        except RuntimeError as e:
            warn('Failed to load weights file {} because:\n{}'.format(config.weight_load_path, e))
    return model

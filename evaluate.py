# -*- coding: utf-8 -*-
# @Time    : 2019/01/10 07:53
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : evaluate.py
# @Software: PyCharm

import os
import warnings
import numpy as np
import torch as t
from torch.nn import Module
from collections import defaultdict
from core import Config, get_model, OrdinalLoss, Visualizer, ipdb
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as Tfn
from core.config import Config


def evaluate(model, img_path, config):
    # type: (Module,list(str),Config)->list(list)
    h, w = (224, 224)
    parent_size = [w * 2, h * 4]
    boxes = [[0, 0], [w, 0],
             [0, h], [w, h],
             [0, 2 * h], [w, 2 * h],
             [0, 3 * h], [w, 3 * h]]
    out_size = [h, w]
    with t.no_grad():
        model.eval()
        # input data
        scene = Image.open(img_path).convert(mode='RGB')
        scene = Tfn.resize(scene, parent_size)
        sub_imgs = []
        for i, j in boxes:
            sub_imgs.append(Tfn.to_tensor(Tfn.crop(scene, i, j, h, w)))
        scene = Tfn.resize(scene, out_size)

        batch_sub_img = t.stack(sub_imgs)
        batch_parent_img = Tfn.to_tensor(scene).unsqueeze(0)

        if config.use_gpu:
            with t.cuda.device(0):
                batch_sub_img = batch_sub_img.cuda()
                batch_parent_img = batch_parent_img.cuda()
        print(batch_sub_img.shape)
        print(batch_parent_img.shape)
        # forward
        batch_inter_prob, batch_cover_rate, batch_final_prob = model(batch_sub_img, batch_parent_img)
        batch_cover_rate.squeeze_()

        batch_final_pred = t.argmax(batch_final_prob, dim=-1)
    resutt = batch_final_pred.detach().cpu().numpy()
    return resutt


def main(args):
    print(args)
    config = Config('inference')
    config.gpu_list = [0]
    config.num_gpu = 1
    model = get_model(config)

    # for img_path in args.input:
    img_path = args.input
    print("Prepare to evaluate %s" % img_path)
    result = evaluate(model, img_path, config)
    print("result: ", result)


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('input', type=str, help='Scene Image(s) to evaluate.')
    parse.add_argument('-o', '--output', type=str, help='write evaluation result into this file')
    args = parse.parse_args()

    main(args)

# -*- coding: utf-8 -*-
# @Time    : 2019/1/5 20:37
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : datasets.py
# @Software: PyCharm

import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict
import torch as t
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as Tfn
from torch.utils.data import DataLoader
from core.config import Config
from .createdatacatalog import create_catalog


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class CloudDataset(data.Dataset):
    label2score = [0.0, 0.1, 0.25, 0.75, 1.0, 0.0]

    def __init__(self, catalog_json, img_resize=None):
        # type:(str,list,Optional(tuple,list))->CloudDataset
        assert os.path.isfile(catalog_json), "{} is not exist or not a file".format(catalog_json)
        self.catalog_json = catalog_json
        with open(catalog_json, 'r') as fp:
            self.scene_list = json.load(fp)

        if _isArrayLike(img_resize):
            w, h = img_resize[0:2]
        elif isinstance(img_resize, int):
            w, h = (img_resize, img_resize)
        else:
            w, h = (224, 224)
        self.width, self.height = w, h
        self.boxes = [[0, 0], [w, 0],
                      [0, h], [w, h],
                      [0, 2 * h], [w, 2 * h],
                      [0, 3 * h], [w, 3 * h]]
        #     self.transform = T.Compose([T.Resize(img_resize), T.ToTensor()])
        # else:
        #     self.transform = T.ToTensor()

    def __getitem__(self, index):
        scene = self.scene_list[index]
        # scene_id = scene["id"]
        sublist = scene["subimages"]
        sub_dict = {}
        for sub in sublist:
            sub_idx = sub["index"]
            sub_label = sub["label"]
            sub_path = sub["path"]
            sub_img = Image.open(sub_path)
            sub_dict[sub_idx] = [sub_img, sub_label]
        sub_tensors = []
        label_s = []
        parent_img = Image.new('rgb', [self.width * 2, self.height * 4])
        label_p = 0
        for i, (img, label) in sub_dict.items():
            parent_img.paste(img, self.boxes[i])
            label_p += self.label2score[label]
            sub_tensors.append(Tfn.to_tensor(img))
            label_s.append(label)
        tensor_p = Tfn.to_tensor(parent_img)
        label_p = t.tensor(label_p)
        tensor_s = t.cat(sub_tensors)
        label_s = t.tensor(label_s)
        return tensor_s, label_s, tensor_p, label_p

    def __len__(self):
        return len(self.scene_list)


def CloudDataLoader(data_type, config):
    # type:(str,Config)->DataLoader
    assert data_type in ["train", "training", "val", "validation", "inference"]
    if data_type in ["train", "training"]:
        data_path = create_catalog(config.train_data_root, config, config.train_catalog_json)
        shuffle = config.shuffle_train
        drop_last = config.drop_last_train
    elif data_type in ["val", "validation", "inference"]:
        data_path = create_catalog(config.val_data_root, config, config.val_catalog_json)
        shuffle = config.shuffle_val
        drop_last = config.drop_last_val
    dataset = CloudDataset(data_path, config.classes_list, config.image_resize)
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)

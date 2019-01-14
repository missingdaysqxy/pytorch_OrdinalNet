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
from createdatacatalog import create_catalog


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
            h, w = img_resize[0:2]
        elif isinstance(img_resize, int):
            h, w = (img_resize, img_resize)
        else:
            h, w = (224, 224)
        self.parent_init_size = [w * 2, h * 4]
        self.boxes = [[0, 0], [w, 0],
                      [0, h], [w, h],
                      [0, 2 * h], [w, 2 * h],
                      [0, 3 * h], [w, 3 * h]]
        self.out_size = [h, w]

    def __getitem__(self, scene_index):
        try:
            scene = self.scene_list[scene_index]
            # scene_id = scene["id"]
            sublist = scene["sublist"]
            sub_dict = {}
            for sub in sublist:
                sub_idx = sub["index"]
                sub_label = sub["label"]
                sub_path = sub["path"]
                sub_img = Image.open(sub_path)
                sub_img = Tfn.resize(sub_img, self.out_size)
                sub_dict[sub_idx] = [sub_img, sub_label]
            for i in range(8):
                if i not in sub_dict.keys():
                    sub_dict[i] = [None, 5]
        except KeyError as e:
            raise ValueError("Invalid key '{}' for {}".format(e, self.catalog_json))
        sub_tensors = []
        label_s = []
        parent_img = Image.new('RGB', self.parent_init_size)
        label_p = 0
        # combine sub-images into parent-image
        for i, (img, label) in sub_dict.items():
            if img is not None:
                parent_img.paste(img, self.boxes[i])
                sub_tensors.append(Tfn.to_tensor(img))
            else:
                sub_tensors.append(t.zeros(3, 224, 224))
            label_p += self.label2score[label]
            label_s.append(label)
        tensor_s = t.cat(sub_tensors)
        label_s = t.tensor(label_s)
        tensor_p = Tfn.to_tensor(Tfn.resize(parent_img, self.out_size))
        label_p = t.tensor(label_p)
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
    dataset = CloudDataset(data_path, config.image_resize)
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)

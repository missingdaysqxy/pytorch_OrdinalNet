# -*- coding: utf-8 -*-
# @Time    : 2019/1/13/013 23:27 下午
# @Author  : qixuan
# @Email   : qixuan.lqx@qq.com
# @File    : createdatacatalog.py
# @Software: PyCharm

import os
import json
from core import Config
from collections import defaultdict


def create_catalog(datadir, config: Config, save_path=None, print_sample=False, integral_limit=4):
    if save_path is None:
        dir, name = os.path.split(datadir)
        save_path = os.path.join(dir, name + "_catalog.json")
    if os.path.exists(save_path) and not config.reload_data:
        print("Find exist catalog '{}' for '{}'".format(save_path, datadir))
        return save_path
    img_paths, parent_ids, labels = step1_getpath(datadir, config.classes_list)
    if print_sample:
        print(img_paths[0:10])
        print(parent_ids[0:10])
        print(labels[0:10])
    print("Found {} sub-images from {}".format(len(img_paths), datadir))

    img_dict, intact_stat = step2_findparent(img_paths, parent_ids, labels)
    print("integrality statistics (total %d):" % len(img_dict.keys()))
    for k, v in intact_stat:
        print("%d sub-images:    %d scenes" % (k, v))

    scene_list = step3_dict2list(img_dict, integral_limit)
    print("Total scenes : %d in intergraliy limit of %d" % (len(scene_list), integral_limit))
    if print_sample:
        print([x["id"] for x in scene_list[0:100]])
        print([x["id"] for x in scene_list[-100:]])
        print(scene_list[0:1])

    with open(save_path, 'w+') as f:
        json.dump(scene_list, f)
    print("Catalog for {} saved into {}".format(datadir, save_path))
    return save_path


def step1_getpath(datadir, classes_list):
    img_paths, parent_ids, labels = [], [], []
    for i, annotation in enumerate(classes_list):
        dir = os.path.join(datadir, annotation)
        fs = os.listdir(dir)
        parent_id = [int(os.path.basename(f).split('_')[0]) for f in fs]
        fs = [os.path.abspath(os.path.join(dir, item)) for item in fs]
        img_paths.extend(fs)
        parent_ids.extend(parent_id)
        labels.extend([i] * len(fs))

    return img_paths, parent_ids, labels


def step2_findparent(img_paths, parent_ids, labels):
    img_dict = defaultdict(dict)
    for path, parent, label in zip(img_paths, parent_ids, labels):
        img_dict[parent][path] = label
    # statistic
    intact_stat = defaultdict(int)
    for i, (k, v) in enumerate(img_dict.items()):
        intact_stat[len(v)] += 1
    return img_dict, sorted(intact_stat.items(), key=lambda item: item[0], reverse=True)


def step3_dict2list(img_dict, integral_limit):
    scenelist = []
    scenelist.clear()
    for id, subimg in img_dict.items():
        sublist = []
        if len(subimg) < integral_limit:
            continue
        for subpath, sublabel in subimg.items():
            basename = os.path.basename(subpath)
            idx = int(basename[basename.index('_') + 1])
            sublist.append({"index": idx, "label": sublabel, "path": subpath})
        sublist.sort(key=lambda x: x["index"])
        scenelist.append({"id": id, "sublist": sublist})
    scenelist = sorted(scenelist, key=lambda x: int(x["id"]), reverse=False)
    return scenelist


def main():
    config = Config(mode="inference")  # switch mode in 'train' and 'inference' for respective data
    datadir = config.train_data_root
    savepath = create_catalog(datadir, config)
    # print("Created training data catalog in: ", savepath)

    datadir = config.val_data_root
    savepath = create_catalog(datadir, config)
    # print("Created validation data catalog in: ", savepath)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @Time    : 2019/1/5 20:37
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : datasets.py
# @Software: PyCharm

import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from model.config import Config


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class CloudDataset(data.Dataset):
    def __init__(self, data_dir, classes_list, img_resize=None):
        # type:(str,list,Optional(tuple,list))->CloudDataset
        assert os.path.isdir(data_dir), "{} is not exist or not a directory".format(data_dir)
        self.data_dir = data_dir
        self.classes_list = classes_list
        self.num_classes = len(self.classes_list)
        for class_name in self.classes_list:
            assert os.path.isdir(os.path.join(data_dir, class_name)), 'class %s not exist in %s' % (
                class_name, data_dir)
        self.img_paths = []
        self.labels = []
        self.scene_names = []
        for i, annotation in enumerate(self.classes_list):
            dir = os.path.join(self.data_dir, annotation)
            fs = os.listdir(dir)
            scene = [os.path.basename(f).split('_')[0] for f in fs]
            fs = [os.path.join(dir, item) for item in fs]
            self.img_paths.extend(fs)
            self.scene_names.extend(scene)
            self.labels.extend([i] * len(fs))
        if _isArrayLike(img_resize) or isinstance(img_resize, int):
            self.transform = T.Compose([T.Resize(img_resize), T.ToTensor()])
        else:
            self.transform = T.ToTensor()

    def __getitem__(self, index):
        path = self.img_paths[index]
        label = self.labels[index]
        scene_name = self.scene_names[index]
        img = Image.open(path)
        img = self.transform(img)
        return img, label, scene_name

    def __len__(self):
        return len(self.img_paths)


def CloudDataLoader(data_type, config):
    # type:(str,Config)->DataLoader
    assert data_type in ["train", "training", "val", "validation", "inference"]
    if data_type in ["train", "training"]:
        data_path = config.train_data_root
        shuffle = config.shuffle_train
        drop_last = config.drop_last_train
    elif data_type in ["val", "validation", "inference"]:
        data_path = config.val_data_root
        shuffle = config.shuffle_val
        drop_last = config.drop_last_val
    dataset = CloudDataset(data_path, config.classes_list, config.image_resize)
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)


if __name__ == "__main__":
    import argparse
    from time import time
    from model import Config
    from cv2 import imshow, waitKey, destroyAllWindows

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_dir', default=r'I:\ordinal_clouds\separate_relabel\train')
    args = parse.parse_args()

    start = time()
    config = Config()
    config.batch_size = 10
    dataset = CloudDataset(args.data_dir, config.classes_list)
    cost_time = time() - start
    print('Cost {} to prepare {} items in dataset {}'.format(cost_time, len(dataset), args.data_dir))
    dataloader = DataLoader(dataset, config.batch_size, True)
    for i, input in enumerate(dataloader):
        # input data
        batch_img, batch_label, batch_scene = input
        print("\n")
        print("shape of images:", batch_img.shape)
        print("labels:", batch_label)
        print("scenes:", batch_scene)
        for i in range(config.batch_size):
            imshow(str(batch_label[i].numpy()), batch_img[i].numpy().astype(np.uint8))
            waitKey(1)
    destroyAllWindows()


import functools
import os
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode
from datasets.transform_custom import *

import torch.nn.functional as F
def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.img_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        self.rgb_transform = transforms.Compose([
                transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # 1. 获取原始大小
        w, h = img.size
        original_shape = (h, w)

        # 2. 暴力查找文件名
        name = str(idx) # 默认值
        
        # 常见的存储文件名的变量名
        potential_attrs = ['ids', 'names', 'image_path', 'img_list', 'imgs', 'samples']
        
        found = False
        for attr in potential_attrs:
            if hasattr(self.dataset, attr):
                data_list = getattr(self.dataset, attr)
                # 确保列表长度足够
                if len(data_list) > idx:
                    item = data_list[idx]
                    # 如果是元组 (path, class_index)，取 path
                    if isinstance(item, tuple):
                        item = item[0]
                    
                    # 提取文件名
                    if isinstance(item, str):
                        name = os.path.splitext(os.path.basename(item))[0]
                        found = True
                        break
        
        # 【调试代码】如果不显示名字，请取消下面这两行的注释，看看终端打印了什么属性
        # if not found and idx == 0:
        #     print(f"DEBUG: Dataset attributes: {dir(self.dataset)}")

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask),
            'inp_rgb': self.rgb_transform(img),
            'name': name,            
            'shape': original_shape 
        }
    
    # def __getitem__(self, idx):
    #     img, mask = self.dataset[idx]
    #     w, h = img.size
    #     # 我们保存为 (H, W) 格式，方便后续处理
    #     original_shape = (h, w)
    #     name = str(idx) # 默认使用索引
         
    #     # 尝试从底层 dataset 中查找常见的文件名列表变量
    #     if hasattr(self.dataset, 'ids'):
    #         name = self.dataset.ids[idx]
    #     elif hasattr(self.dataset, 'names'):
    #         name = self.dataset.names[idx]
    #     elif hasattr(self.dataset, 'image_path'):
    #          # 如果存的是路径列表
    #         name = os.path.splitext(os.path.basename(self.dataset.image_path[idx]))[0]
        
    #     # 确保 name 是字符串，且不带后缀
    #     if isinstance(name, str):
    #         name = os.path.splitext(name)[0]


    #     return {
    #         'inp': self.img_transform(img),
    #         'gt': self.mask_transform(mask),
    #         'inp_rgb': self.rgb_transform(img),
    #         'name': name,           
    #         'shape': original_shape 
    #     }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.transform = transforms.Compose([
                RandomHorizontalFlip(),
                RandomScaleCrop(base_size=self.inp_size, crop_size=self.inp_size),
                RandomGaussianBlur(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor(),
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        # random filp
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        img = transforms.Resize((1280, 1280))(img)
        mask = transforms.Resize((1280, 1280), interpolation=InterpolationMode.NEAREST)(mask)
        sample = {"image": img, "label": mask}
        sample = self.transform(sample)
        return {
            'inp': sample["image"],
            'gt': sample["label"][None]
        }
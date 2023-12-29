import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
import os
from PIL import Image
import random
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

import copy


datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    name = dataset_spec['name']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset

class Resampler(torch.nn.Module):

    def __init__(self, inp_size, interpolation_mode, resampling_factor):
        super(Resampler, self).__init__()
        self.inp_size = inp_size
        self.resampling_factor = resampling_factor
        self.interpolation_mode = InterpolationMode(interpolation_mode)
    
    def forward(self, img):
        new_size = self.inp_size // self.resampling_factor

        downsampler = transforms.Resize(size=(new_size,new_size), interpolation=self.interpolation_mode)
        upsampler = transforms.Resize(size=(self.inp_size, self.inp_size), interpolation=self.interpolation_mode)
        
        downsampled_image = downsampler.forward(img)
        transformed_image = upsampler.forward(downsampled_image)

        return transformed_image


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, interpolation_mode="nearest", resampling_factor = 1, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.img_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                Resampler(inp_size = inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                     std=[1, 1, 1])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((self.inp_size, self.inp_size)),
                Resampler(inp_size = inp_size, interpolation_mode=interpolation_mode, resampling_factor=resampling_factor),
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

        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        mask = transforms.Resize((self.inp_size, self.inp_size), interpolation=InterpolationMode.NEAREST)(mask)

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask)
        }
    

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = make(spec['dataset'])
    dataset = make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True)
    return loader

import yaml
inp_size = 1024
interpolation_mode = 'bicubic'
resampling_factor = 1
config_file = "/home/kandelaki/git/SAM-Adapter-PyTorch/visualize/visualize.yaml"

# load config 
with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

train_loader = make_data_loader(config.get('train_dataset'), tag='train')
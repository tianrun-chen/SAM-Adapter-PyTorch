""" MIT License

Copyright (c) 2023 tianrun-chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 """

import os
import sys

#adding path to the some modules
for modules in ['datasets', 'models', 'utils', 'sod_metric']:
    path = os.path.abspath(modules)
    sys.path.append(path)

import argparse

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, output_dir):
    model.eval()

    pbar = tqdm(loader, leave=False, desc='fw')
    nr = 0
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        cpu_pred = pred.cpu()
        vector_temp = cpu_pred.detach().squeeze().numpy()

        filepath = loader.dataset.dataset.files[nr]
        nr += 1
        last = filepath.split('/')[-1]
        file_name = last.split('.')[0]
        save_path_img = f'{output_dir}/png'
        save_path_np = f'{output_dir}/numpy'
        try:
            np.save(f'{save_path_np}/{file_name}.npy',vector_temp)
            
        except:
            os.makedirs(save_path_np)
            np.save(f'{save_path_np}/{file_name}.npy',vector_temp)
        try:
            plt.imsave(f'{save_path_img}/{file_name}.png',vector_temp, cmap=cm.gray)
        except:
            os.makedirs(save_path_img)
            plt.imsave(f'{save_path_img}/{file_name}.png',vector_temp, cmap=cm.gray)


if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config')
        parser.add_argument('--model')
        parser.add_argument('--output_dir', default='output')
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        spec = config['fw_dataset']
        dataset = datasets.make(spec['dataset'])
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            num_workers=8)
    
        model = models.make(config['model']).cuda()
        sam_checkpoint = torch.load(args.model, map_location='cuda:0')
        model.load_state_dict(sam_checkpoint, strict=True)
        
        eval_psnr(loader, model, args.output_dir)
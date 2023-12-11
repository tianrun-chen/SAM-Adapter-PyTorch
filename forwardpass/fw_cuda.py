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

import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


'''def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)'''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, output_dir):
    model.eval()

    pbar = tqdm(loader, leave=False, desc='val')
    nr = 0
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        cpu_pred = pred.cpu()
        vector_temp = cpu_pred.detach().squeeze().numpy()

        filepath = loader.dataset.dataset.dataset_1.files[nr]
        nr += 1
        last = filepath.split('/')[-1]
        file_name = last.split('.')[0]
        save_path_img = f'{output_dir}/dv'
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


def run_forwardpass(config, model, output_dir):
    with torch.no_grad():
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        spec = config['test_dataset']
        dataset = datasets.make(spec['dataset'])
        dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                            num_workers=8)
    
        model = models.make(config['model']).cuda()
        sam_checkpoint = torch.load(model, map_location='cuda:0')
        model.load_state_dict(sam_checkpoint, strict=True)
        
        eval_psnr(loader, model, output_dir)
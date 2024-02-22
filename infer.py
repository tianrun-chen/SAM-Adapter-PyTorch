import argparse
import os

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
from torchvision import transforms
import metric
import writer
import logging
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms


class Infer:
    def __init__(self, model, loader, save_path=None, inp_size=1024):
        self.model = model
        self.inp_size = inp_size
        self.save_path = save_path
        self.loader = loader
        os.makedirs(self.save_path, exist_ok=True)
    
    def compute(self, threshold=0.3):
        self.model.eval()


        for i, batch in enumerate(self.loader):
            for k, v in batch.items():
                batch[k] = v.to(self.model.device)

            inp = batch['inp']

            pred = torch.sigmoid(self.model.infer(inp))
            pred = (pred>threshold).float()
            
            pred = transforms.ToPILImage()(pred.squeeze(0).float())
            pred.save(os.path.join(self.save_path, f'pred_{i}.png'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam-vit-b.yaml")
    parser.add_argument('--model', default=None)
    parser.add_argument('--save-path', default="infer_results")
    args = parser.parse_args()

    save_path = args.save_path

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    model = models.make(config['model'])
    device = model.device
    model = model.to(device)

    if model.device == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False
        )
        model = model.module

    sam_checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=False)


    loader = utils.make_data_loader(config, 'infer_dataset')
    
    infer = Infer(model, loader, save_path=save_path)

    pred = infer.compute()
    
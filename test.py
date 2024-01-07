import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import writer  
from torchvision import transforms
from mmcv.runner import load_checkpoint
import metric
from PIL import Image
import matplotlib.pyplot as plt
from datasets.resample_transform import Resampler

class Test:
    
        def __init__(self, model, test_loader, save_path, resampler, original_image_dataset):
            self.model = model
            self.test_loader = test_loader
            self.save_path = save_path
            self.resampler = resampler
            self.original_image_dataset = original_image_dataset
    
            self.metrics = metric.Metric()
            self.writer = writer.Writer(os.path.join(self.save_path, 'test'))
    
        def start(self):
    
            self.model.eval()
    
            pbar = tqdm(total=len(self.test_loader), leave=False, desc='test')
   
            for i, batch in enumerate(self.test_loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.model.device)
    
                self.metrics.reset_metrics()

                inp = batch['inp']
                gt = batch['gt']
                gt = (gt>0)
    
                pred = torch.sigmoid(self.model.infer(inp))
    
                values = self.metrics.update_and_compute(pred, gt)
                
                self.writer.write_metrics_and_means(values, i)
                self.writer.write_pr_curve(pred,gt, i)
                self.writer.write_gt_vs_pred_figure(pred, gt, i, "Gt vs Pred")

                original_image  = self.original_image_dataset[i]["inp"]
                original_image = self.original_image_dataset.inverse_transform(original_image)

                resampled = self.original_image_dataset.inverse_transform(inp)

                self.writer.write_resampled_vs_orig_figure(resampled, original_image, i, "Resampled vs Orig")
                self.writer.write_overlay_mask_figure(resampled, pred, gt, i, "Overlay Mask")

                if pbar is not None:
                    pbar.update(1)
    
            if pbar is not None:
                pbar.close()

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sam-vit-b.yaml')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--save-path', default="./save/")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # Save config
        with open(os.path.join(args.save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)
    
    model = models.make(config['model'])
    device = model.device
    model = model.to(device)

    sam_checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    resampling_spec = config["test_dataset"]["wrapper"]["args"]
    resampler = Resampler(resampling_spec["inp_size"], resampling_spec["interpolation_mode"], resampling_spec["resampling_factor"])

    # Load original image dataset (without any resampling)
    spec = config['test_dataset']
    config["test_dataset"]["wrapper"]["args"]["resampling_factor"] = 1
    original_image_dataset = datasets.make(spec['dataset'])
    original_image_dataset = datasets.make(spec['wrapper'], args={'dataset': original_image_dataset})

    test = Test(model, loader, args.save_path, resampler, original_image_dataset)
    test.start()


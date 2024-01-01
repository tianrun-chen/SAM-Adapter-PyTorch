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
from mmcv.runner import load_checkpoint
import metric
from PIL import Image
import matplotlib.pyplot as plt

# TODO OOP this and use writer wrapper

def write_metrics(values, means, i):
    jaccard, dice, accuracy, precision, recall, specificity = values
    writer.add_scalar('jaccard', jaccard, global_step=i)
    writer.add_scalar('dice', dice, global_step=i)
    writer.add_scalar('accuracy', accuracy, global_step=i)
    writer.add_scalar('precision', precision, global_step=i)
    writer.add_scalar('recall', recall, global_step=i)
    writer.add_scalar('specificity', specificity, global_step=i)

def create_gt_vs_pred_figure(gt, pred, threshold=0.5):
    gt = 1 - gt
    pred = 1 - pred
    pred = pred > threshold
    gt = transforms.ToPILImage()(gt.squeeze(0).float())
    pred = transforms.ToPILImage()(pred.squeeze(0).float())
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(gt)
    ax1.set_title('Ground Truth')
    ax2.imshow(pred)
    ax2.set_title('Prediction')
    return fig


def test(loader, model):
    
    device = model.device
    model.eval()
    
    pbar = tqdm(loader, leave=False, desc='val')
    
    for i, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.to(device)

        metrics.reset()

        inp = batch['inp']
        gt = batch['gt']
        gt = (gt>0).int()
        pred = torch.sigmoid(model.infer(inp))

        metrics.update(pred, gt)
        metric_values = metrics.compute_values()

        write_metrics(metric_values, None, i)
        writer.add_pr_curve('PR Curve', gt, pred, global_step=i)
        if i % 5 == 0:
            fig = create_gt_vs_pred_figure(gt, pred)
            writer.add_figure('Ground Truth vs Prediction', fig, global_step=i)

     
     

if __name__ == '__main__':
    global log, writer, metrics
    metrics = metric.Metric()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    log, writer = utils.set_save_path(os.path.join("save","test"), remove=False)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)
    
    global inverse_transform
    inverse_transform = dataset.inverse_transform

    model = models.make(config['model'])
    device = model.device
    
    model = model.to(device)
    sam_checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    test(loader, model)

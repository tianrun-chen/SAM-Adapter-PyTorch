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

def batched_predict(model, inp, coord, bsize):
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
    return toPIL(tensor)

def write_metrics(values, means, i):
    jaccard, dice, accuracy, precision, recall, specificity = values
    writer.add_scalar('jaccard', jaccard, global_step=i)
    writer.add_scalar('dice', dice, global_step=i)
    writer.add_scalar('accuracy', accuracy, global_step=i)
    writer.add_scalar('precision', precision, global_step=i)
    writer.add_scalar('recall', recall, global_step=i)
    writer.add_scalar('specificity', specificity, global_step=i)

def create_overlay_plot(inp, mask):
    inp = tensor2PIL(inp.squeeze(0))
    mask = tensor2PIL(mask.squeeze(0))

    mask = mask.convert('RGB')
    out = Image.blend(inp, mask, 0.5)
    return transforms.ToTensor()(out)


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
        metric_values = metrics.compute_values(pred, gt)
        write_metrics(metric_values, None, i)
        writer.add_pr_curve('PR Curve', gt, pred, global_step=i)
        if i % 5 == 0:
            inversed = inverse_transform(inp)
            plot_pred = create_overlay_plot(inversed, pred)
            plot_gt = create_overlay_plot(inversed, gt)
            writer.add_image('Prediction Overlay', plot_pred, global_step=i)
            writer.add_image('Ground Truth Overlay', plot_gt, global_step=i)
     
     

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

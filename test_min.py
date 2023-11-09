import argparse
import os
import numpy as np

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

import matplotlib.cm as cm
import time


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    nr = 0

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))
        import matplotlib.pyplot as plt
        vector_temp = pred.detach().squeeze().numpy()

        #split_list = img_list[nr].split('.')
        #file_name = split_list[0]
        filepath = loader.dataset.dataset.dataset_1.files[nr]
        last = filepath.split('/')[-1]
        file_name = last.split('.')[0]
        save_path_img = 'test/dv'
        save_path_np = 'test/numpy'
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
        
        nr += 1
        print(f'max: {vector_temp.max()}')
        print(f'min: {vector_temp.min()}')
        print(f'mean: {vector_temp.mean()}')
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    #getting the filenames for saving. --> assuming the dataloader loads it the same way
    #img_list = os.listdir(spec['dataset']['args']['root_path_1'])

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model'])
    sam_checkpoint = torch.load(args.model)
    model.load_state_dict(sam_checkpoint, strict=False)

    start_time = time.time()

    metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    end_time = time.time()
    print(f'Total elapsed Time: {end_time-start_time} s for predicting {len(loader.dataset.dataset.dataset_1.files)} images. \n This makes {(end_time-start_time)/len(loader.dataset.dataset.dataset_1.files)} s/img ')
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))

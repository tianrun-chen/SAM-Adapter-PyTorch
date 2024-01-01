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

def init_process_group():
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    # train_loader = DataLoader(datasets.wrappers.TrainDataset(configs..  dataset, batch_size=spec['batch_size'],
    #     shuffle=False, num_workers=8, pin_memory=True, sampler=sampler))
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    
    sampler = None
    if device == 'cuda':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader



def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model'])
        model.to(model.device)
        
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model'])
        model.to(model.device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    return model, optimizer, epoch_start, lr_scheduler




def eval(loader, model):
    model.eval()

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    
    for i, batch in enumerate(loader):
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        batch_pred = []
        batch_gt = []

        if model.device == 'cuda':
            batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

            dist.all_gather(batch_pred, pred)
            dist.all_gather(batch_gt, batch['gt'])
        else:
            batch_pred = pred
            batch_gt = batch['gt']
        
        batched_gt = (batched_gt>0).int()
        metrics.update(batch_pred, batch_gt)
        metric_values = metrics.compute_values()
        
        writer_wrapper.write_metrics(metric_values, None, i)
        writer_wrapper.write_pr_curve(batch_pred, batch_gt, i)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

def train(train_loader, model):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()

        if model.device == 'cuda': 
            batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
            dist.all_gather(batch_loss, model.loss_G)
        else:
            batch_loss = [torch.zeros_like(model.loss_G)]
        
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer_, log_info, writer_wrapper

    # TODO merge writers, OOP for train and test
    writer_wrapper = writer.Writer(os.path.join(save_path, 'train'))
    
    config = config_
    log, writer_ = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    if model.device == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False
        )
        model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):

        eval(val_loader, model)

        if model.device == 'cuda':
            train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer_.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer_.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
           
                # TODO save best model according to the metric
                eval(val_loader, model)
                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer_.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    global local_rank, device, metrics, writer_

    metrics = metric.Metric()
    local_rank = args.local_rank
    device = config['model']['args']['device']
    
    if  device == 'cuda':
            init_process_group()
    else:
        local_rank = 0
        print("Config loaded.")

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)

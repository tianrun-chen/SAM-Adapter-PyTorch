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

class Train:

    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader, epoch_start, epoch_max, epoch_val, save_path, local_rank):
        self.model = model
        self.config = config
        self.local_rank = local_rank
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epoch_start = epoch_start
        self.epoch_max = epoch_max
        self.epoch_val = epoch_val
        self.save_path = save_path

        self.writer = writer.Writer(os.path.join(self.save_path, 'train'))
        self.validation_metric = metric.Metrics(['JaccardIndex', 'DiceCoefficient', 'Precision', 'Recall', 'Accuracy', 'F1Score', 'AUCROC'], device=model.device)


    def eval(self, epoch=None):

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm(total=len(self.val_loader), leave=False, desc='val')
        else:
            pbar = None

        # Reset metrics (mean and current) for this epoch
        self.validation_metric.reset()
        metric_values = None

        for i, batch in enumerate(self.val_loader):
            for k, v in batch.items():
                batch[k] = v.to(self.model.device)

            inp = batch['inp']

            pred = torch.sigmoid(self.model.infer(inp))

            batch_pred = []
            batch_gt = []

            if self.model.device == 'cuda':
                batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
                batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

                dist.all_gather(batch_pred, pred)
                dist.all_gather(batch_gt, batch['gt'])
            else:
                batch_pred = pred
                batch_gt = batch['gt']
            
            batch_gt = (batch_gt>0).int()

            self.validation_metric.reset_current()
            self.validation_metric.update(batch_pred, batch_gt)
            metric_values = self.validation_metric.compute()

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.writer.write_means(metric_values, epoch)
        
        mean_IoU = metric_values["JaccardIndex"][1]
        
        return mean_IoU.item()

    def train(self):

        self.model.train()

        if self.local_rank == 0:
            pbar = tqdm(total=len(self.train_loader), leave=False, desc='train')
        else:
            pbar = None

        loss_list = []
        
        for batch in self.train_loader:
            for k, v in batch.items():
                batch[k] = v.to(self.model.device)
            
            inp = batch['inp']
            gt = batch['gt']
            self.model.set_input(inp, gt)
            self.model.optimize_parameters()

            if self.model.device == 'cuda': 
                batch_loss = [torch.zeros_like(self.model.loss_G) for _ in range(dist.get_world_size())]
                dist.all_gather(batch_loss, self.model.loss_G)
            else:
                batch_loss = [torch.zeros_like(self.model.loss_G)]
                batch_loss[0] = self.model.loss_G
            
            loss_list.extend(batch_loss)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        loss = [i.item() for i in loss_list]
        return mean(loss)

    def start(self):

        best_mean_IoU = -1

        for epoch in range(epoch_start, epoch_max + 1):
            
            if self.model.device == 'cuda':
                self.train_loader.sampler.set_epoch(epoch)
            train_loss_G = self.train()
            self.lr_scheduler.step()

            if self.local_rank == 0:
                logging.info('Epoch: ' + str(epoch)+ '/' + str(self.epoch_max) + ' train_loss_G: ' + str(train_loss_G))
                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Training_loss', train_loss_G, epoch)

                self.save('last')

            if (self.epoch_val is not None) and (epoch % self.epoch_val == 0):
            
                    
                    current_mean_IoU = self.eval(epoch)

                    if current_mean_IoU > best_mean_IoU:
                        best_mean_IoU = current_mean_IoU
                        if self.local_rank == 0:
                            logging.info('Epoch: ' + str(epoch)+ '/' + str(self.epoch_max) + ' val_mean_IoU: ' + str(current_mean_IoU))
                            self.save('best')

            self.writer.flush()
    
    def save(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, f"model_epoch_{name}.pth"))

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam-vit-b.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    save_path = None
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # Save config
        
        save_path = config['write_dir']
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
            
    os.makedirs(config.get('log_dir'), exist_ok=True)
    logging.basicConfig(filename=os.path.join(config.get('log_dir'),"log.txt"), level=logging.INFO, format="%(asctime)s %(message)s")
    
    local_rank = args.local_rank
    device = config['model']['args']['device']

    if  device == 'cuda':
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        print("Config loaded.")
    
    model, optimizer, epoch_start, lr_scheduler = utils.prepare_training(config)
    
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
        logging.info('model_grad_params:' + str(model_grad_params) + '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
    }

    train_loader, val_loader = utils.make_data_loaders(config=config)
    
    train = Train(model, optimizer, lr_scheduler, train_loader, val_loader, epoch_start, epoch_max, epoch_val, save_path, local_rank=local_rank)
    train.start()
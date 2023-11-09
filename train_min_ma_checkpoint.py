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

import shutil
from PIL import Image
import copy

import psutil
from datetime import datetime

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    
    if True:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

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

    pbar = tqdm(total=len(loader), leave=False, desc='val')

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        pred_list.append(pred)
        gt_list.append(batch['gt'])
        pbar.update(1)

    pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training():
    if config.get('resume') is not None:
        print('resume training')
        #MA-Load checkpoint
        #checkpoint = torch.load(config.get('resume').get('checkpoint'))
        #model.load_state_dict(checkpoint['model'])
        model = models.make(config['model'])
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model'])
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    model.train()

    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    loss_list = []


    for index, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        model.optimize_parameters()
        loss_list.append(model.loss_G.item())
        #ma save loss
        try:
            with open(os.path.join('logs','loss_list.txt'),'a') as f:
                f.write(f'Batch loss: {model.loss_G.item()} \n')
        except:
            os.makedirs('logs')
            with open(os.path.join('logs','loss_list.txt'),'a') as f:
                f.write(f'Batch loss: {model.loss_G.item()} \n')
        #ma log system
        with open(os.path.join('logs','system_logs.txt'),'a') as f:
                f.write(f'{datetime.now()}: CPU: {psutil.cpu_percent(4)} RAM: {psutil.virtual_memory()[2]}\n') 
        pbar.update(1)

    pbar.close()

    loss = mean(loss_list)

    return loss

    
def main_training(config_, save_path,args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    ##changed writing to txt instead of yaml because always crashed in second iteration
    #with open(os.path.join(save_path,'config.txt'),'w') as f:
        #f.write(str(config))
    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)

        #ma save loss
        with open(os.path.join('logs','loss_list.txt'),'a') as f:
            f.write(f'Mean loss over dv({save_path}): {train_loss_G}\n')

        lr_scheduler.step()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()

        save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
                eval_type=config.get('eval_type'))

            log_info.append('val: {}={:.4f}'.format(metric1, result1))
            writer.add_scalars(metric1, {'val': result1}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric2, result2))
            writer.add_scalars(metric2, {'val': result2}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric3, result3))
            writer.add_scalars(metric3, {'val': result3}, epoch)
            log_info.append('val: {}={:.4f}'.format(metric4, result4))
            writer.add_scalars(metric4, {'val': result4}, epoch)

            if config['eval_type'] != 'ber':
                if result1 > max_val_v:
                    max_val_v = result1
                    save(config, model, save_path, 'best')
            else:
                if result3 < max_val_v:
                    max_val_v = result3
                    save(config, model, save_path, 'best')

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

            log(', '.join(log_info))
            writer.flush()


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

#MA trainings divider
def get_folder_nr(folder_path):
    split_path = folder_path.split('v')
    return int(split_path[-1])

def check_image(filepath):
    try:
        im = Image.open(filepath)
        im.verify()
        im.close()
        return True
    except:
        return False
    

def img_mask_ok(img_path, mask_path, filename):
    if filename.lower().endswith(('.png','.jpg','.jpeg')):
        filepath_img = f'{img_path}/{filename}'
        filepath_mask = f'{mask_path}/{filename}'
        if check_image(filepath_img):
            if check_image(filepath_mask):
                return True

        os.remove(filepath_mask)
        os.remove(filepath_img)
        print('corrupted image found')
    return False


def divider(img_path, mask_path):
    part = 0
    lst_new_img_paths = []
    lst_new_mask_paths = []
    count = 40
    i = 0    

    for filename in os.listdir(img_path):
        if img_mask_ok(img_path, mask_path, filename):
            if i <= 0:
                new_path_img = os.path.join(img_path,f'dv{part}')
                new_path_masks = os.path.join(mask_path,f'dv{part}')
                try:
                    os.mkdir(new_path_img)
                    os.mkdir(new_path_masks)
                except:
                    print('divide folder already exists')
                lst_new_img_paths.append(new_path_img)
                lst_new_mask_paths.append(new_path_masks)

                #number for next folder
                part += 1
                i = count

            shutil.move(f'{img_path}/{filename}',f'{new_path_img}/{filename}')
            shutil.move(f'{mask_path}/{filename}',f'{new_path_masks}/{filename}')
            i -= 1
        elif 'dv' in filename:
            new_path_img = os.path.join(img_path,filename)
            new_path_masks = os.path.join(mask_path,filename)
            lst_new_img_paths.append(new_path_img)
            lst_new_mask_paths.append(new_path_masks)

    lst_new_img_paths.sort(key=get_folder_nr)
    lst_new_mask_paths.sort(key=get_folder_nr)
    return lst_new_img_paths, lst_new_mask_paths


def main(config, save_path, resume_e, resume_dv, args):
    train_config = config.get('train_dataset')
    img_path = train_config.get('dataset').get('args').get('root_path_1')
    mask_path = train_config.get('dataset').get('args').get('root_path_2')

    new_img_path, new_mask_path = divider(img_path,mask_path)
    print(f'training will be resumed at Epoch: {resume_e} and dv: {resume_dv}')
    epochen = 30

    for j in tqdm(range(resume_e,epochen), desc='Total Epochs'):
        for i in tqdm(range(resume_dv, len(new_img_path)), desc='Total all Trainingsdata'):
            #to start at 0 after the first resume
            resume_dv = 0
            config['train_dataset']['dataset']['args']['root_path_1'] = new_img_path[i]
            config['train_dataset']['dataset']['args']['root_path_2'] = new_mask_path[i]
            #for first round in new epoch
            if i == 0 and j > 0:
                model_path = os.path.join(save_path,f'dv_{j-1}_{len(new_img_path)-1}','model_epoch_last.pth')
                config['sam_checkpoint'] = model_path 
            #change model path to last training div
            if i > 0:
                model_path = os.path.join(save_path,f'dv_{j}_{i-1}','model_epoch_last.pth')
                config['sam_checkpoint'] = model_path
            new_save_path = os.path.join(save_path,f'dv_{j}_{i}')
            try:
                os.makedirs(new_save_path)
            except:
                print('save folder already exists, data may be overwritten')
            print(f'started training round {i} in epoch {j}')
            running_config = copy.deepcopy(config)
            main_training(running_config, new_save_path, args)

#MA trainings divider


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    #MA Argument
    parser.add_argument('--resume_e', type=int, default=0)
    parser.add_argument('--resume_dv', type=int, default=0)
    args = parser.parse_args()

    #MA
    e_resume = args.resume_e
    dv_resume = args.resume_dv

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
        #if local_rank == 0:
        #    print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    gl_save_path = os.path.join('./save', save_name)

    main(config, gl_save_path, e_resume, dv_resume, args=args)
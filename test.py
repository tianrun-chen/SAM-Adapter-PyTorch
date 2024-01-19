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
    
        def __init__(self, model, test_loader, trained_on_dataset, save_path, original_image_dataset, trained_on_factor, tested_on_factor):
            self.model = model
            self.test_loader = test_loader
            self.save_path = save_path
            self.trained_on_dataset = trained_on_dataset
            self.original_image_dataset = original_image_dataset
            self.trained_on_factor = trained_on_factor
            self.tested_on_factor = tested_on_factor
            self.metrics = metric.Metrics(['JaccardIndex', 'DiceCoefficient', 'Precision', 'Recall', 'Accuracy', 'F1Score', 'AUCROC'], device=model.device)
            self.writer = writer.Writer(os.path.join(self.save_path, 'test'))
    
        def start(self):
    
            self.model.eval()
    
            pbar = tqdm(total=len(self.test_loader), leave=False, desc='test')
   
            for i, batch in enumerate(self.test_loader):
                for k, v in batch.items():
                    batch[k] = v.to(self.model.device)
    
            
                inp = batch['inp']
                gt = batch['gt']
                gt = (gt>0)
    
                pred = torch.sigmoid(self.model.infer(inp))
                
                self.metrics.reset_current()
                self.metrics.update(pred, gt)
                values = self.metrics.compute()

                self.writer.write_metrics_and_means(values, i)
                self.writer.write_pr_curve(pred,gt, i)
                self.writer.write_gt_vs_pred_figure(pred, gt, i, "Gt vs Pred")
            
                original_image  = self.original_image_dataset[i]["inp"]
                original_image = self.original_image_dataset.inverse_transform(original_image)

                tested_on_image = self.original_image_dataset.inverse_transform(inp)

                trained_on_image = self.trained_on_dataset[i]["inp"]
                trained_on_image = self.trained_on_dataset.inverse_transform(trained_on_image)

                self.writer.write_trained_on_vs_tested_on_vs_original(trained_on_image, tested_on_image, original_image, self.trained_on_factor, self.tested_on_factor, i, "Trained on vs Tested on vs Original")
                self.writer.write_overlay_confusion_matrix_figure(tested_on_image, pred, gt, values, i, "Overlay Confusion Matrix")
                
                if pbar is not None:
                    pbar.update(1)
    
            if pbar is not None:
                pbar.close()

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sam-vit-b.yaml')
    parser.add_argument('--model', help="Path to the trained model checkpoint")
    parser.add_argument('--prompt', default='none')
    parser.add_argument('--dataset', default='val_dataset')
    args = parser.parse_args()



    save_path = None
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        save_path = config['write_dir']
        os.makedirs(save_path, exist_ok=True)
        # Save config
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    
    
    os.makedirs(save_path, exist_ok=True)

    dataset_to_use = args.dataset

    # Create Testing Dataset and its loader
    spec = config[dataset_to_use]
    testing_dataset = datasets.make(spec['dataset'])
    testing_dataset = datasets.make(spec['wrapper'], args={'dataset': testing_dataset})
    loader = DataLoader(testing_dataset, batch_size=spec['batch_size'],
                        num_workers=0)
    
    model = models.make(config['model'])
    device = model.device
    model = model.to(device)

    sam_checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    # Create Training Dataset
    trained_on_factor = config["train_dataset"]["wrapper"]["args"]["resampling_factor"]
    config[dataset_to_use]["wrapper"]["args"]["resampling_factor"] = trained_on_factor
    trained_on_dataset = datasets.make(spec['dataset'])
    trained_on_dataset = datasets.make(spec['wrapper'], args={'dataset': trained_on_dataset})



    # Load original image dataset (without any resampling)
    spec = config[dataset_to_use]
    config[dataset_to_use]["wrapper"]["args"]["resampling_factor"] = 1
    original_image_dataset = datasets.make(spec['dataset'])
    original_image_dataset = datasets.make(spec['wrapper'], args={'dataset': original_image_dataset})

    test = Test(model, loader, trained_on_dataset, save_path,  original_image_dataset, trained_on_factor, trained_on_factor)
    test.start()


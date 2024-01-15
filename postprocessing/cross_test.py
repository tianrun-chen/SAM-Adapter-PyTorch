# Tests all the trained models on resampled data with 0.2 step size in resampling factor
# Example: 
# A model trained on factor 2 resampled images will be tested on factor 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, ..., 6.0 resampled images
# This for all the trained models 
from test import Test
import yaml
import os
import datasets
from torch.utils.data import DataLoader
from datasets.resample_transform import Resampler
import models
import torch
import argparse
def test_this_model(trained_model, trained_model_factor, config, dataset, factor_step_size):
    
    save_path = os.path.join("cross_test", trained_model_factor, dataset, str("on_" + factor_step_size))
    
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        os.makedirs(save_path, exist_ok=True)
        # Save config
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    

    dataset_to_use = dataset
    # change the resampling factor for the tested dataset (test_dataset or val_dataset)
    config[dataset]["wrapper"]["args"]["resampling_factor"] = factor_step_size
    spec = config[dataset_to_use]
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)
    
    model = models.make(config['model'])
    device = model.device
    model = model.to(device)

    sam_checkpoint = torch.load(trained_model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    resampling_spec = config[dataset_to_use]["wrapper"]["args"]
    resampler = Resampler(resampling_spec["inp_size"], resampling_spec["interpolation_mode"], resampling_spec["resampling_factor"])

    # Load original image dataset (without any resampling)
    spec = config[dataset_to_use]
    config[dataset_to_use]["wrapper"]["args"]["resampling_factor"] = 1
    original_image_dataset = datasets.make(spec['dataset'])
    original_image_dataset = datasets.make(spec['wrapper'], args={'dataset': original_image_dataset})

    test = Test(model, loader, save_path, resampler, original_image_dataset)
    test.start()

def test_all_models(models, config, factor_step_size=0.2, range_start=1.0, range_end=6.0):
    for dataset in ["val_dataset", "test_dataset"]:
        for factor in range(range_start, range_end, factor_step_size):
            for model in models:
                test_this_model(model, config, dataset, factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/sam-vit-b.yaml")
    parser.add_argument('--models', default=None, help = "Path to seperate folders containing the models to be tested (Split with comma)")
    args = parser.parse_args()
    
    models = args.models.split(",")
    test_all_models(models, args.config)

# Tests all the trained models on resampled data with 0.2 step size in resampling factor
# Example: 
# A model trained on factor 2 resampled images will be tested on factor 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, ..., 6.0 resampled images
# This for all the trained models 
from test import Test
import yaml
import os
import datasets
from torch.utils.data import DataLoader
import models
import torch
import argparse
import numpy as np
from tqdm import tqdm

def test_this_model(trained_model, trained_on_factor, config, dataset_to_use, testing_factor):
    
    save_path = os.path.join("cross_test", "trained_on_"+str(trained_on_factor), dataset_to_use, "tested_on_" + str(testing_factor))
    
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        os.makedirs(save_path, exist_ok=True)
        # Save config
        with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # Create Testing Dataset and its loader
    config[dataset_to_use]["wrapper"]["args"]["resampling_factor"] = testing_factor
    spec = config[dataset_to_use]
    testing_dataset = datasets.make(spec['dataset'])
    testing_dataset = datasets.make(spec['wrapper'], args={'dataset': testing_dataset})

    loader = DataLoader(testing_dataset, batch_size=spec['batch_size'],
                        num_workers=8)
    
    model = models.make(config['model'])
    device = model.device
    model = model.to(device)

    sam_checkpoint = torch.load(trained_model, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    
    # Create Training Dataset
    spec = config[dataset_to_use]
    config[dataset_to_use]["wrapper"]["args"]["resampling_factor"] = trained_on_factor
    trained_on_dataset = datasets.make(spec['dataset'])
    trained_on_dataset = datasets.make(spec['wrapper'], args={'dataset': trained_on_dataset})

    
    # Create Original Image Dataset
    spec = config[dataset_to_use]
    config[dataset_to_use]["wrapper"]["args"]["resampling_factor"] = 1
    original_image_dataset = datasets.make(spec['dataset'])
    original_image_dataset = datasets.make(spec['wrapper'], args={'dataset': original_image_dataset})

     

    test = Test(model, loader, trained_on_dataset, save_path,  original_image_dataset, trained_on_factor, testing_factor)
    test.start()
    

def test_all_models(models_meta_data, config, factor_step_size=0.2, range_start=2.4, range_end=6.0):
    for meta_data in models_meta_data:
        model = meta_data.split(";")[0]
        for dataset in ["val_dataset", "test_dataset"]:
            for testing_factor in tqdm(np.arange(range_start, range_end, factor_step_size), desc="Testing " + model + " on " + dataset + " with different resampling factors"):
                trained_on_factor = meta_data.split(";")[1]
                test_this_model(model, int(trained_on_factor), config, dataset, testing_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, help = "Path to config file")
    parser.add_argument('--models', default=None, help = "Path to seperate folders containing the models to be tested (Split with comma) and the factors they where trained on (Seperated by ;)")
    args = parser.parse_args()
    
    models_meta_data = args.models.split(",")

    test_all_models(models_meta_data, args.config)

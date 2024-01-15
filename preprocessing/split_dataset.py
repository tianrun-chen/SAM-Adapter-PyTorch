import torch.utils.data as data
import torch
import os
import argparse
import shutil

def make_dirs(path):
    for split_type in ["train", "eval", "test"]:
        os.makedirs(os.path.join(path, split_type), exist_ok=True)


def main(tif_folder, seed, output):
    make_dirs(output)
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]
    tif_files.sort()
    num_files = len(tif_files)
    train_split = int(0.8 * num_files)
    eval_split = int(0.1 * num_files)
    test_split = num_files - train_split - eval_split        
    generator = torch.Generator().manual_seed(seed)

    train, eval, test = data.random_split(tif_files, [train_split, eval_split, test_split], generator=generator)
    
    # save splits
    for split_type, split in zip(["train", "eval", "test"], [train, eval, test]):
        for file in split:
            file_path = os.path.join(tif_folder, file)
            shutil.copy(file_path, os.path.join(output, split_type))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif-folder', default="./split_folder_empty_removed")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random train-eval-test set generator")
    parser.add_argument("--output", default="./random_train_eval_test_split", help="Train, eval and test split output folder")
    args = parser.parse_args()
    tif_folder = args.tif_folder
    seed = args.seed
    output = args.output

    main(tif_folder, seed, output)

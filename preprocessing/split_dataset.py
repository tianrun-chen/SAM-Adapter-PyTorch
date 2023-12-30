from datasets.image_folder import PairedImageFolders
import torch.utils.data as data
import torch
import os
import argparse
from tqdm import tqdm

def make_dirs(path):
    for split_type in ["train", "eval", "test"]:
        os.makedirs(os.path.join(path, split_type), exist_ok=False)

def save_split(split, path_img, path_mask):
    for i, (img, mask) in enumerate(split):
        img.save(os.path.join(path_img, f"{i}.png"))
        mask.save(os.path.join(path_mask, f"{i}.png"))

def save(train,eval,test, output):
    for split in  tqdm([("train", train), ("eval",eval), ("test",test)], desc="Saving training, evaluation and testing splits"):
        save_split(split[1], os.path.join(output,"img", split[0]), os.path.join(output,"masks", split[0]))

def main(image_folder, mask_folder, seed, output):
    
    make_dirs(os.path.join(output,"img"))
    make_dirs(os.path.join(output,"masks"))
    
    paired_folders = PairedImageFolders(image_folder, mask_folder)
   
    # Same generator for both image and label folders
    generator1 = torch.Generator().manual_seed(seed)
    dataset_size = len(paired_folders)
    train_cnt, eval_cnt, test_cnt =  (int(dataset_size * 0.8), int(dataset_size * 0.1), int(dataset_size * 0.1))
    train, eval, test = data.random_split(paired_folders, [train_cnt, eval_cnt, test_cnt], generator=generator1)
    
    save(train,eval,test, output)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-folder', default="../temp/images_preprocessed_split")
    parser.add_argument("--mask-folder", default="../temp/masks_preprocessed_split")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random train-eval-test set generator")
    parser.add_argument("--output", default="../load", help="Output folder path (load) for img and mask folders")
    args = parser.parse_args()
    image_folder = args.images_folder
    mask_folder = args.mask_folder
    seed = args.seed
    output = args.output

    main(image_folder, mask_folder, seed, output)

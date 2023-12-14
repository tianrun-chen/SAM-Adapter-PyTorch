import sys
import os

#adding path to the some modules
for modules in ['preprocessing', 'postprocessing']:
    path = os.path.abspath(modules)
    sys.path.append(path)

from download_open_data import DownloadOpenData
from split_return import Split
from save_img_mask import Save
from ma_make_overlay import *
import argparse

def run(lat_1, lon_1, lat_2, lon_2, output_folder):
    download_ = DownloadOpenData()
    download_.wgs84_download(lat_1, lon_1, lat_2, lon_2, f'{output_folder}/tiles_download')

    split = Split(2500, 256)
    images = split.splitImages(f'{output_folder}/tiles_download')

    save_ = Save()
    save_.saveImg(f'{output_folder}/split_img', images, '')

    os.system(f'python fw_cuda.py --config configs/ma_B_cuda.yaml --model save/_ma_B/dv_29_18/model_epoch_last.pth --output_dir {output_folder}/pred_masks')

    run_overlay(f'{output_folder}/split_img', f'{output_folder}/pred_masks/png', f'{output_folder}/overlay')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat_1', type=float)
    parser.add_argument('--lon_1', type=float)
    parser.add_argument('--lat_2', type=float)
    parser.add_argument('--lon_2', type=float)
    parser.add_argument('--output_folder', default='forwardpass/data')

    args = parser.parse_args()

    lat_1 = args.lat_1
    lon_1 = args.lon_1
    lat_2 = args.lat_2
    lon_2 = args.lon_2
    output_folder = args.output_folder

    run(lat_1, lon_1, lat_2, lon_2, output_folder)
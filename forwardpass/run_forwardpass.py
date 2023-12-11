import sys
import os
#adding path to the preprocessing module
path = os.path.abspath('preprocessing')
sys.path.append(path)

from download_open_data import DownloadOpenData
from split_return import Split
from save_img_mask import Save
import fw_cuda
import argparse

#under construction

def run(lat_1, lon_1, lat_2, lon_2, output_folder):
    download_ = DownloadOpenData()
    download_.wgs84_download(lat_1, lon_1, lat_2, lon_2, output_folder)

    split = Split(2500, 256)
    images = split.splitImages(output_folder)

    save_ = Save()
    save_.saveImg('forwardpass/data/split_img', images, '')

    fw_cuda.run_forwardpass('configs/ma_B_cuda_fw.yaml', 'save/_ma_B/dv_29_18/model_epoch_last.pth', 'output')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat_1', type=float)
    parser.add_argument('--lon_1', type=float)
    parser.add_argument('--lat_2', type=float)
    parser.add_argument('--lon_2', type=float)
    parser.add_argument('--output_folder', default='forwardpass/data/tiles_download')

    args = parser.parse_args()

    lat_1 = args.lat_1
    lon_1 = args.lon_1
    lat_2 = args.lat_2
    lon_2 = args.lon_2
    output_folder = args.output_folder

    run(lat_1, lon_1, lat_2, lon_2, output_folder)
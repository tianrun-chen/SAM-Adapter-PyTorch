import argparse
import os

from masks import MaskMaker
from split_return import Split
from save_img_mask import Save
from np2png_ma import Np2Png
from rename_union_new_ma import RenameUnion
from proof_not_empty import Proof
from train_test_split_ma import TrainTestSplit
import create_folderstructure

class Run:
    def runAll(geotif, geojson):
        create_folderstructure.create_folders('all')

        for geojson_file in os.listdir(geojson):
            filename = geojson_file.split('.')[0]
            maskmaker = MaskMaker(f'{geojson}/{geojson_file}',geotif,filename)
            maskmaker.process()

            split = Split(2500,512)
            images = split.splitImages(f"data/masked_images_{filename}")
            masks = split.splitMask(f"data/{filename}_masks")

            save_ = Save()
            save_.saveImg('data/rdy/img', images, filename)
            save_.saveMask('data/rdy/masks', masks, filename)


        proof = Proof('data/rdy/img', 'data/rdy/masks')
        proof.is_empty_del()

        test_train_split = TrainTestSplit
        test_train_split.split('data/rdy/img','load/img')

        test_train_split.split('data/rdy/masks','load/masks')
    
    def runSplit(split_images, split_masks, geojson):
        create_folderstructure.create_folders('all')

        for geojson_file in os.listdir(geojson):
            filename = geojson_file.split('.')[0]

            split = Split(2500,512)
            images = split.splitImages(split_images)
            masks = split.splitMask(split_masks)

            save_ = Save()
            save_.saveImg('data/rdy/img', images, filename)
            save_.saveMask('data/rdy/masks', masks, filename)


        proof = Proof('data/rdy/img', 'data/rdy/masks')
        proof.is_empty_del()

        test_train_split = TrainTestSplit
        test_train_split.split('data/rdy/img','load/img')

        test_train_split.split('data/rdy/masks','load/masks')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtype', default='all')
    parser.add_argument('--geotif', default=None)
    parser.add_argument('--geojson', default=None)
    parser.add_argument('--splitimages', default=None)
    parser.add_argument('--splitmasks', default=None)
    args = parser.parse_args()

    run_type = args.runtype
    geotif = args.geotif
    geojson = args.geojson
    split_images = args.splitimages
    split_masks = args.splitmasks

    if run_type == 'all':
            Run.runAll(geotif, geojson)
    elif run_type == 'split':
            Run.runSplit(split_images, split_masks, geojson)
    else:
        print('Please enter correct --runtype all/split')
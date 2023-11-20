from PIL import Image
import numpy as np
import os

class Save:
    def saveImg(self, folder_path, images, json_file):
        if not os.path.isdir(folder_path): os.makedirs(folder_path)
        for image in images:
            image[0].save(f'{folder_path}/{image[1]}_{image[2]}_{json_file}.png')


    def saveMask(self, folder_path, masks, json_file):
        if not os.path.isdir(folder_path): os.makedirs(folder_path)
        for mask in masks:
            #conv to binary img
            im_bin = mask[0] > 0
            im = Image.fromarray(im_bin)
            im.save(f'{folder_path}/{mask[1]}_{mask[2]}_{json_file}.png')
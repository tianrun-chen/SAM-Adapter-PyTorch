from PIL import Image
import numpy as np
import os


class Np2Png:

    def __init__(self, src_path, dest_path):
        self.src_path_ = src_path
        self.dest_path_ = dest_path

    def np_2_png(self):
        for filename in os.listdir(self.src_path_):
            A = np.load(f"{self.src_path_}/{filename}")
            #conv to binary img
            im_bin = A > 0
            im = Image.fromarray(im_bin)
            #remove unessesary extrensions
            split_filename = filename.split('.')
            core_filename = split_filename[0]
            try:
                im.save(f"{self.dest_path_}/{core_filename}.png")
            except:
                os.mkdir(self.dest_path_)
                im.save(f"{self.dest_path_}/{core_filename}.png")
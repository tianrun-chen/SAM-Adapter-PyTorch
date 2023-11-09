import os
from PIL import Image 
import numpy as np

class Proof:

    def __init__(self, img_path, mask_path):
        self.img_path_ = img_path
        self.mask_path_ = mask_path

    def is_empty_del(self):
        deleted_images = 0
        total_images = 0
        for mask_filename in os.listdir(self.mask_path_):
            mask_filepath = os.path.join(self.mask_path_, mask_filename)
            mask_png = Image.open(mask_filepath)
            mask_np = np.asarray(mask_png)
            if mask_np.sum() < 20:
                os.remove(mask_filepath)
                img_filepath = os.path.join(self.img_path_,mask_filename)
                os.remove(img_filepath)
                print(f'deleted image {mask_filename} total deleted images {deleted_images} of {total_images}')
                deleted_images += 1
            total_images += 1
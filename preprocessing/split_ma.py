from PIL import Image
import numpy as np
import os
from pathlib import Path

#only tested with 2500x2500 to 1024x1024 images
class Split:
    def __init__(self, in_size = 2500, dest_size = 1024):
        self.dest_size_ = dest_size
        self.in_size_ = in_size
        self.split_coos_ = self.calcSplit(self.in_size_)

    
    def calcSplit(self, width_height):
        #only one calculation because working with squared formats
        #calculating the overlap on each side (half_overlap)
        t = width_height//self.dest_size_
        left = width_height%self.dest_size_
        overlap = self.dest_size_-left
        half_overlap = overlap//2

        split_coos = []

        for r in range(t+1):
           for c in range(t+1):
               #xmin,ymin,xmax,ymax
               split_coos.append((c*self.dest_size_-c*half_overlap,r*self.dest_size_-r*half_overlap,(c+1)*self.dest_size_-c*half_overlap,(r+1)*self.dest_size_-r*half_overlap))
        
        return split_coos
    

    def splitImages(self, image_filepath):
        #get input folder
        split_image_filepath = image_filepath.split("/")
        split_folder = Path("data")/"split"
        image_folder = Path("data")/"split"/split_image_filepath[-1]
        if not split_folder.exists(): split_folder.mkdir()
        if not image_folder.exists(): image_folder.mkdir()
        for filename in os.listdir(image_filepath):
            im = Image.open(f"{image_filepath}/{filename}")
            q = 0
            for coos in self.split_coos_:
                image = im.crop(coos)
                split_filename = filename.split('.')
                image_name = split_filename[0]
                image.save(f"{image_folder}/{image_name}_{q}.png")
                q += 1
    
    def splitMask(self, mask_filepath):
        #get input folder
        split_mask_filepath = mask_filepath.split("/")
        #to be saved in...
        split_folder = Path("data")/"split"
        mask_folder = Path("data")/"split"/split_mask_filepath[-1]
        if not split_folder.exists(): split_folder.mkdir()
        if not mask_folder.exists(): mask_folder.mkdir()
        for filename in os.listdir(mask_filepath):
            mask = np.load(f"{mask_filepath}/{filename}")

            q = 0
            for coos in self.split_coos_:
                new_mask = mask[coos[1]:coos[3],coos[0]:coos[2]]
                split_filename = filename.split('.')
                mask_name = split_filename[0]
                np.save(mask_folder/f"{mask_name}_{q}.npy",new_mask)
                q += 1

    

    def splitBoundingBox(self, bounding_box, image_id):

        new_bounding_box = {}
        q = 0
        for coos in self.split_coos_:
            #possibility that box between new image tiles --> only in the image which contains the whole box could be changed?
            if bounding_box[0] > coos[0] and bounding_box[1] > coos[1] and bounding_box[2] < coos[2] and bounding_box[3] < coos[3]:
                  new_bounding_box[f"{image_id}_{q}"] = (bounding_box[0]-coos[0],bounding_box[1]-coos[1],bounding_box[2]-coos[0],bounding_box[3]-coos[1])
            q += 1
        return new_bounding_box
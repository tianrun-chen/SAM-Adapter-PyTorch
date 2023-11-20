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
        images = []
        for filename in os.listdir(image_filepath):
            im = Image.open(f"{image_filepath}/{filename}")
            q = 0
            for coos in self.split_coos_:
                image_name = filename.split('.')[0]
                image = im.crop(coos)
                images.append((image,image_name,q))
                q += 1
        
        return images
    
    def splitMask(self, mask_filepath):
        masks = []
        for filename in os.listdir(mask_filepath):
            mask = np.load(f"{mask_filepath}/{filename}")
            q = 0
            for coos in self.split_coos_:
                mask_name = filename.split('.')[0]
                new_mask = mask[coos[1]:coos[3],coos[0]:coos[2]]
                masks.append((new_mask,mask_name,q))
                q += 1
        
        return masks

    

    def splitBoundingBox(self, bounding_box, image_id):

        new_bounding_box = {}
        q = 0
        for coos in self.split_coos_:
            #possibility that box between new image tiles --> only in the image which contains the whole box could be changed?
            if bounding_box[0] > coos[0] and bounding_box[1] > coos[1] and bounding_box[2] < coos[2] and bounding_box[3] < coos[3]:
                  new_bounding_box[f"{image_id}_{q}"] = (bounding_box[0]-coos[0],bounding_box[1]-coos[1],bounding_box[2]-coos[0],bounding_box[3]-coos[1])
            q += 1
        return new_bounding_box
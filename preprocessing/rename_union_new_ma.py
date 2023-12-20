import os
from PIL import Image


class RenameUnion:

    def __init__(self, src_path, dest_path, incl_info):
        self.src_path_ = src_path
        self.dest_path_ = dest_path
        self.incl_info_ = incl_info

    def rename_union(self):
        for filename in os.listdir(self.src_path_):
            dot_split = filename.split(".")
            image_name = dot_split[0]
            im = Image.open(f"{self.src_path_}/{filename}")
            try:
                im.save(f"{self.dest_path_}/{image_name}_{self.incl_info_}.png")
            except:
                os.makedirs(self.dest_path_)
                im.save(f"{self.dest_path_}/{image_name}_{self.incl_info_}.png")
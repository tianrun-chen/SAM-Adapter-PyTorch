#code by Aravinda https://aravinda-gn.medium.com/how-to-split-image-dataset-into-train-validation-and-test-set-5a41c48af332 

import os
import random
import shutil


class TrainTestSplit:

    def split(src_path, dst_path):
        # path to destination folders
        train_folder = os.path.join(dst_path, 'train')
        val_folder = os.path.join(dst_path, 'eval')
        test_folder = os.path.join(dst_path, 'test')

        # Define a list of image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        # Create a list of image filenames in 'data_path'
        imgs_list = [filename for filename in os.listdir(src_path) if os.path.splitext(filename)[-1] in image_extensions]

        # Sets the random seed 
        random.seed(42)

        # Shuffle the list of image filenames
        random.shuffle(imgs_list)

        # determine the number of images for each set
        train_size = int(len(imgs_list) * 0.85)
        val_size = int(len(imgs_list) * 0.05)
        test_size = int(len(imgs_list) * 0.1)

        # Create destination folders if they don't exist
        for folder_path in [train_folder, val_folder, test_folder]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # Copy image files to destination folders
        for i, f in enumerate(imgs_list):
            if i < train_size:
                dest_folder = train_folder
            elif i < train_size + val_size:
                dest_folder = val_folder
            else:
                dest_folder = test_folder
            shutil.copy(os.path.join(src_path, f), os.path.join(dest_folder, f))
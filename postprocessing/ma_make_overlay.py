import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def run_overlay(img_folder_path, mask_folder_path, overlay_folder_path):
    for filename in os.listdir(mask_folder_path):
        mask_array = img2np(f'{mask_folder_path}/{filename}')
        img_array = img2np(f'{img_folder_path}/{filename}')
        mask_array = cv2.resize(mask_array, (256, 256))
        mask_array = mask_array[:,:,0:1]
        overlayed = overlay(img_array, mask_array, (0,255,0), 0.4)
        clean_filename = cleanFilename(filename)
        try:
            plt.imsave(f'{overlay_folder_path}/{clean_filename}.png',overlayed)
        except:
            os.makedirs(overlay_folder_path)
            plt.imsave(f'{overlay_folder_path}/{clean_filename}.png',overlayed)

def cleanFilename(filename):
    split_list = filename.split('.')
    clean_filename = split_list[0]
    return clean_filename

def img2np(image_file):
    img = Image.open(image_file)
    img.load()
    np_array = np.asarray(img, dtype='uint8')
    return np_array

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
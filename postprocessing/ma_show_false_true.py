import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import ma_make_overlay
import os
from statistics import mean


def create_overlay_stats(img_path, test_mask_path, pred_mask_path, save_path):
    iou_list = []
    for img_filename, test_mask_filename, pred_mask_filename in zip(os.listdir(img_path),os.listdir(test_mask_path),os.listdir(pred_mask_path)):
        print(img_path)
        print(img_filename)
        image = Image.open(f'{img_path}/{img_filename}')

        test_mask_img = Image.open(f'{test_mask_path}/{test_mask_filename}')
        test_mask = np.asarray(test_mask_img)


        pred_mask_img= Image.open(f'{pred_mask_path}/{pred_mask_filename}')
        pred_mask_img= pred_mask_img.resize((np.shape(test_mask)))
        pred_mask = np.asarray(pred_mask_img)

        output = np.zeros(shape=(np.shape(test_mask)[0],np.shape(test_mask)[1],4),dtype=np.uint8)

        true_zero = 0
        true_one = 0
        false_zero = 0
        false_one = 0
        sum_pixels = np.shape(test_mask)[0]*np.shape(test_mask)[1]


        for row in range(len(test_mask)):
            for col in range(len(test_mask)):
                if pred_mask[row,col] == test_mask[row,col] and test_mask[row,col] == False:
                    #black
                    output[row,col] = [0,0,0,128]
                    true_zero += 1
                elif pred_mask[row,col] == test_mask[row,col] and test_mask[row,col] == True:
                    #yellow
                    output[row,col] = [255,255,0,128]
                    true_one += 1
                #false positive
                elif pred_mask[row,col] > test_mask[row,col]:
                    #green
                    output[row,col] = [0,255,0,128]
                    false_one += 1
                #false negative
                elif pred_mask[row,col] < test_mask[row,col]:
                    #blue
                    output[row,col] = [0,0,255,128]
                    false_zero += 1

        iou = true_one/(false_one+false_zero+true_one)
        iou_list.append(iou)
        mean_iou = mean(iou_list)
        print(f' true Zero: {true_zero/sum_pixels}\n true One: {true_one/sum_pixels}\n total True: {(true_zero+true_one)/sum_pixels}\n false Zero: {false_zero/sum_pixels}\n false One: {false_one/sum_pixels}\n Total false: {(false_zero+false_one)/sum_pixels}')
        print(f'Predicted size difference: {np.sum(pred_mask)/np.sum(test_mask)}')
        print(f'IoU: {iou}')
        print(f'mean IoU: {mean_iou} after {len(iou_list)} images')

        alpha = 0.2
        img_conv = image.convert('RGBA')
        output_img = Image.fromarray(output,'RGBA')
        image_combined = Image.blend(img_conv,output_img,alpha)

        clean_filename = ma_make_overlay.cleanFilename(img_filename)

        plt.imshow(image_combined)
        plt.title(f'yellow: positive, green: false positive, blue: false negative\nPredicted size difference: {np.sum(pred_mask)/np.sum(test_mask)}\nIoU: {true_one/(false_one+false_zero+true_one)}')
        plt.savefig(f'{save_path}/{clean_filename}.png')



create_overlay_stats('load/img/test','load/masks/test','binary','overlay/stats_overlay')
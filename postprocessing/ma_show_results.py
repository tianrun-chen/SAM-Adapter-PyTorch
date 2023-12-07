import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from statistics import mean
import argparse

def create_overlay_stats(img_path, test_mask_path, pred_mask_path, save_path, threshold):
    iou_list = []
    for filename in os.listdir(img_path):
        filename = filename.split('.')[0]
        print(img_path)
        print(filename)
        image = Image.open(f'{img_path}/{filename}.png')

        test_mask_img = Image.open(f'{test_mask_path}/{filename}.png')
        test_mask = np.asarray(test_mask_img)


        pred_mask_np= np.load(f'{pred_mask_path}/{filename}.npy')
        bin_mask_np = pred_mask_np > threshold
        bin_mask_img = Image.fromarray(bin_mask_np)
        pred_mask_img= bin_mask_img.resize((np.shape(test_mask)))
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


        plt.imshow(image_combined)
        plt.title(f'yellow: positive, green: false positive, blue: false negative\nPredicted size difference: {np.sum(pred_mask)/np.sum(test_mask)}\nIoU: {true_one/(false_one+false_zero+true_one)}')
        plt.savefig(f'{save_path}/{filename}.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagepath', default='load/img/test')
    parser.add_argument('--groundtruth', default='load/masks/test')
    parser.add_argument('--prednp', default='test/numpy')
    parser.add_argument('--savepath', default='overlay/stats_overlay')
    parser.add_argument('--threshold', default=0.5, type=float)

    args = parser.parse_args()

    image_path = args.imagepath
    groundtruth = args.groundtruth
    pred_np = args.prednp
    save_path = args.savepath
    threshold = args.threshold

    create_overlay_stats(image_path, groundtruth, pred_np, save_path, threshold)
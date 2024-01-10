# Writes logs to the Tensorboard
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
class Writer:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def create_overlay_mask_figure(self, image, pred, gt, threshold=0.5):
        # remove batch dimension
        image = image.squeeze(0)
        pred = pred.squeeze(0)
        pred = pred.squeeze(0)
       
        # Image with ground truth mask
        image_gt = torch.clone(image)
        gt = gt.squeeze(0)
        gt = gt.squeeze(0)

        image_gt[0][gt] = 0
        image_gt[1][gt] = 1
        image_gt[2][gt] = 0
       
        pred = pred > threshold
        # Set pixels where mask is true to red
        image[0][pred] = 1
        image[1][pred] = 0
        image[2][pred] = 0


        image = transforms.ToPILImage()(image.float())
        image_gt = transforms.ToPILImage()(image_gt.float())

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax1.imshow(image_gt)
        ax1.set_title('Ground Truth Overlay')
        ax2.imshow(image)
        ax2.set_title('Prediction Overlay')
        return fig
    
    def create_gt_vs_pred_figure(self, pred, gt, threshold=0.5):
        pred = pred > threshold
        gt = transforms.ToPILImage()(gt.squeeze(0).float())
        pred = transforms.ToPILImage()(pred.squeeze(0).float())
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax1.imshow(gt)
        ax1.set_title('Ground Truth')
        ax2.imshow(pred)
        ax2.set_title('Prediction')
        return fig
    
    def create_resampled_vs_orig_figure(self, resampled, original):
        resampled = transforms.ToPILImage()(resampled.squeeze(0).float())
        original = transforms.ToPILImage()(original.squeeze(0).float())
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        
        ax1.imshow(resampled)
        ax1.set_title('Resampled')
        ax2.imshow(original)
        ax2.set_title('Original')
        return fig

    def write_metrics_and_means(self, values, step):
        current_dice, mean_dice = values["DiceCoefficient"]
        current_jaccard, mean_jaccard = values["JaccardIndex"]
        self.writer.add_scalars('Dice', {'current': current_dice, 'mean': mean_dice}, global_step=step)
        self.writer.add_scalars('JaccardIndex', {'current': current_jaccard, 'mean': mean_jaccard}, global_step=step)


    def write_pr_curve(self, pred, gt, step):
        self.writer.add_pr_curve('PR Curve', gt, pred, global_step=step)

    def write_figure(self, fig, step, desc):
        self.writer.add_figure(desc, fig, global_step=step)

    def write_gt_vs_pred_figure(self,pred,gt, step, desc):
        fig = self.create_gt_vs_pred_figure(pred, gt)
        self.write_figure(fig, step, desc)

    def write_resampled_vs_orig_figure(self, resampled, orig, step, desc):
        fig = self.create_resampled_vs_orig_figure(resampled, orig)
        self.write_figure(fig, step, desc)

    def write_overlay_mask_figure(self, image, pred, gt, step, desc):
        fig = self.create_overlay_mask_figure(image, pred, gt)
        self.write_figure(fig, step, desc)

    def add_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, global_step=step)

    def flush(self):
        self.writer.flush()
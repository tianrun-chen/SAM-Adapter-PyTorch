# Writes logs to the Tensorboard
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from torchvision import transforms
class Writer:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def create_overlay_mask_figure(self, image, mask, threshold=0.5):
        # remove batch dimension
        image = image.squeeze(0)
        mask = mask.squeeze(0)
        mask = mask.squeeze(0)
        # Apply threshold to mask
        mask = 1 - mask
        mask = mask > threshold
    
        # Set pixels where mask is true to red
        image[0][mask] = 1
        image[1][mask] = 0
        image[2][mask] = 0

        image = transforms.ToPILImage()(image.float())
        fig = plt.figure()
        fig.set_figheight(15)
        fig.set_figwidth(15)
        plt.imshow(image)
        return fig
    
    def create_gt_vs_pred_figure(self, pred, gt, threshold=0.5):
        gt = 1 - gt
        pred = 1 - pred
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

        (jaccard, mean_jaccard), (dice, mean_dice), (accuracy, mean_accuracy), (precision, mean_precision), (recall, mean_recall), (specificity, mean_specificity) = values
        
        self.writer.add_scalars('Jaccard Index (IoU)', {"Current": jaccard, "Mean": mean_jaccard}, global_step=step)
        self.writer.add_scalars('Dice', {"Current": dice, "Mean": mean_dice}, global_step=step)
        self.writer.add_scalars('Accuracy', {"Current": accuracy, "Mean": mean_accuracy}, global_step=step)
        self.writer.add_scalars('Precision', {"Current": precision, "Mean": mean_precision}, global_step=step)
        self.writer.add_scalars('Recall', {"Current": recall, "Mean": mean_recall}, global_step=step)
        self.writer.add_scalars('Specificity', {"Current": specificity, "Mean": mean_specificity}, global_step=step)


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

    def write_overlay_mask_figure(self, image, mask, step, desc):
        fig = self.create_overlay_mask_figure(image, mask)
        self.write_figure(fig, step, desc)
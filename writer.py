# Writes logs to the Tensorboard
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from PIL import Image
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
    
    def create_overlay_confusion_matrix_figure(self, image_tensor, pred, gt, metrics, threshold=0.5):
        
        canvas = self.create_binary_confusion_matrix_tensor(pred, gt, threshold)
   
        # remove batch dimension
        image_tensor = image_tensor.squeeze(0)

        image = transforms.ToPILImage()(image_tensor)
        image = image.convert("RGBA")

        canvas_image = transforms.ToPILImage()(canvas.float())

        canvas_image = canvas_image.convert("RGBA")
        
        blended = Image.blend(image, canvas_image, alpha = 0.4)

        blended_tensor = transforms.ToTensor()(blended)

        # Ignore true negatives (Not interesting)
        gt = gt.squeeze(0)
        gt = gt.squeeze(0)
        pred = pred.squeeze(0)
        pred = pred.squeeze(0)
        pred = pred > threshold
        tn = torch.logical_and(torch.logical_not(gt), torch.logical_not(pred))

        blended_tensor[0][tn == 1] = image_tensor[0][tn == 1]
        blended_tensor[1][tn == 1] = image_tensor[1][tn == 1]
        blended_tensor[2][tn == 1] = image_tensor[2][tn == 1]

        blended = transforms.ToPILImage()(blended_tensor)

        # create figure out of canvas
        fig, ax = plt.subplots(1, 1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        current_jaccard, _ = metrics["JaccardIndex"]
        current_dice, _ = metrics["DiceCoefficient"]
        current_precision, _ = metrics["Precision"]
        current_recall, _ = metrics["Recall"]
        current_accuracy, _ = metrics["Accuracy"]
        current_f1, _ = metrics["F1Score"]
        current_AUCROC, _ = metrics["AUCROC"]

        # Create Legend for the figure (including the metrics)

        legend_elements = [Patch(facecolor='green', edgecolor='black',
                            label='True Positive'),
                        Patch(facecolor='yellow', edgecolor='black',
                            label='False Positive'),
                        Patch(facecolor='red', edgecolor='black',
                            label='False Negative'),
                        Line2D([0], [0], color='black', lw=4, label=f'Jaccard Index (IoU): {current_jaccard:.4f}'),
                        Line2D([0], [0], color='black', lw=4, label=f'Dice Coefficient: {current_dice:.4f}'),
                        Line2D([0], [0], color='black', lw=4, label=f'Precision: {current_precision:.4f}'),
                        Line2D([0], [0], color='black', lw=4, label=f'Recall: {current_recall:.4f}'),
                        Line2D([0], [0], color='black', lw=4, label=f'Accuracy: {current_accuracy:.4f}'),
                        Line2D([0], [0], color='black', lw=4, label=f'F1 Score: {current_f1:.4f}'),
                        Line2D([0], [0], color='black', lw=4, label=f'AUCROC: {current_AUCROC:.4f}')]
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.imshow(blended)
        ax.set_title('Visual Confusion Matrix')
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
    
    def create_binary_confusion_matrix_tensor(self, pred, gt, threshold=0.5):
        gt = gt.squeeze(0)
        gt = gt.squeeze(0)
        pred = pred.squeeze(0)
        pred = pred.squeeze(0)
        pred = pred > threshold

        tp = torch.logical_and(gt, pred)
        fp = torch.logical_and(torch.logical_not(gt), pred)
        fn = torch.logical_and(gt, torch.logical_not(pred))
        # create canvas
        canvas = torch.zeros_like(tp)

        # extent tp to 3 channels
        canvas = torch.stack([canvas, canvas, canvas], dim = 0)

        # set true positives to green
        canvas[0][tp == 1] = 0
        canvas[1][tp == 1] = 255
        canvas[2][tp == 1] = 0
        # set false positives to yellow
        canvas[0][fp == 1] = 255
        canvas[1][fp == 1] = 255
        canvas[2][fp == 1] = 0
        # set false negatives to red
        canvas[0][fn == 1] = 255
        canvas[1][fn == 1] = 0
        canvas[2][fn == 1] = 0
        # true negatives are black by default
        
        return canvas

    def create_trained_on_vs_tested_on_vs_original(self, trained_on, tested_on, original, trained_on_factor, tested_on_factor):
        
        trained_on = transforms.ToPILImage()(trained_on.squeeze(0).float())
        tested_on = transforms.ToPILImage()(tested_on.squeeze(0).float())
        original = transforms.ToPILImage()(original.squeeze(0).float())
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(25)
        fig.set_figwidth(25)
        
        ax1.imshow(trained_on)
        ax1.set_title('Trained on factor ' + str(trained_on_factor))
        ax2.imshow(tested_on)
        ax2.set_title('Tested on factor ' + str(tested_on_factor))
        ax3.imshow(original)
        ax3.set_title('Original')
        return fig

    def write_metrics_and_means(self, values, step):
        current_dice, mean_dice = values["DiceCoefficient"]
        current_jaccard, mean_jaccard = values["JaccardIndex"]
        current_precision, mean_precision = values["Precision"]
        current_recall, mean_recall = values["Recall"]
        current_accuracy, mean_accuracy = values["Accuracy"]
        current_f1, mean_f1 = values["F1Score"]
        current_AUCROC, mean_AUCROC = values["AUCROC"]

        self.writer.add_scalars('Dice', {'current': current_dice, 'mean': mean_dice}, global_step=step)
        self.writer.add_scalars('IoU', {'current': current_jaccard, 'mean': mean_jaccard}, global_step=step)
        self.writer.add_scalars('Precision', {'current': current_precision, 'mean': mean_precision}, global_step=step)
        self.writer.add_scalars('Recall', {'current': current_recall, 'mean': mean_recall}, global_step=step)
        self.writer.add_scalars('Accuracy', {'current': current_accuracy, 'mean': mean_accuracy}, global_step=step)
        self.writer.add_scalars('F1', {'current': current_f1, 'mean': mean_f1}, global_step=step)
        self.writer.add_scalars('AUCROC', {'current': current_AUCROC, 'mean': mean_AUCROC}, global_step=step)

    def write_means(self, values, step):
        _, mean_dice = values["DiceCoefficient"]
        _, mean_jaccard = values["JaccardIndex"]
        _, mean_precision = values["Precision"]
        _, mean_recall = values["Recall"]
        _, mean_accuracy = values["Accuracy"]
        _, mean_f1 = values["F1Score"]
        _, mean_AUCROC = values["AUCROC"]
        self.writer.add_scalars('Dice', {'mean': mean_dice}, global_step=step)
        self.writer.add_scalars('IoU', {'mean': mean_jaccard}, global_step=step)
        self.writer.add_scalars('Precision', {'mean': mean_precision}, global_step=step)
        self.writer.add_scalars('Recall', {'mean': mean_recall}, global_step=step)
        self.writer.add_scalars('Accuracy', {'mean': mean_accuracy}, global_step=step)
        self.writer.add_scalars('F1', {'mean': mean_f1}, global_step=step)
        self.writer.add_scalars('AUCROC', {'mean': mean_AUCROC}, global_step=step)
        
    def write_pr_curve(self, pred, gt, step):
        self.writer.add_pr_curve('PR Curve', gt, pred, global_step=step)

    def write_figure(self, fig, step, desc):
        self.writer.add_figure(desc, fig, global_step=step)

    def write_gt_vs_pred_figure(self,pred,gt, step, desc):
        fig = self.create_gt_vs_pred_figure(pred, gt)
        self.write_figure(fig, step, desc)

    def write_trained_on_vs_tested_on_vs_original(self, trained_on, tested_on, original, trained_on_factor, tested_on_factor, step, desc):
        fig = self.create_trained_on_vs_tested_on_vs_original(trained_on, tested_on, original, trained_on_factor, tested_on_factor)
        self.write_figure(fig, step, desc)

    def write_overlay_mask_figure(self, image, pred, gt, step, desc):
        fig = self.create_overlay_mask_figure(image, pred, gt)
        self.write_figure(fig, step, desc)

    def write_overlay_confusion_matrix_figure(self, image, pred, gt, metrics, step, desc):
        fig = self.create_overlay_confusion_matrix_figure(image, pred, gt, metrics)
        self.write_figure(fig, step, desc)
    
    def add_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, global_step=step)

    def flush(self):
        self.writer.flush()
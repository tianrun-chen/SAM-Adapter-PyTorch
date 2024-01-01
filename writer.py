# Writes logs to the Tensorboard
from tensorboardX import SummaryWriter
import os
class Writer:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def write_metrics(self, values, means, step):
        jaccard, dice, accuracy, precision, recall, specificity = values
        self.writer.add_scalar('jaccard', jaccard, global_step=step)
        self.writer.add_scalar('dice', dice, global_step=step)
        self.writer.add_scalar('accuracy', accuracy, global_step=step)
        self.writer.add_scalar('precision', precision, global_step=step)
        self.writer.add_scalar('recall', recall, global_step=step)
        self.writer.add_scalar('specificity', specificity, global_step=step)

    def write_pr_curve(self, pred, gt, step):
        self.writer.add_pr_curve('PR Curve', gt, pred, global_step=step)
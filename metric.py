# Metrics for assessing the model
#rom torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryJaccardIndex # IoU
from torchmetrics.classification import Dice 
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinarySpecificity
from torchmetrics.classification import BinaryAccuracy

from torchmetrics.aggregation import MeanMetric

class Metric:
    jaccard = BinaryJaccardIndex()
    dice =  Dice()
    accuracy = BinaryAccuracy()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    specificity = BinarySpecificity()

    metrics = [jaccard, dice, accuracy, precision, recall, specificity]
    means = []
    
    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, pred, target):
        for metric in self.metrics:
            metric.update(pred, target)
    
    def compute_values(self, pred, target):
        self.update(pred, target)

        metric_values = []
        for metric in self.metrics:
            metric_values.append(metric.compute())
        return metric_values
    
    def compute_means(self):
        metric_means = []
        for metric in self.means:
            metric_means.append(metric.compute())
        return metric_means
        
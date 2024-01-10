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
    # jaccard = BinaryJaccardIndex()
    # dice =  Dice()
    # accuracy = BinaryAccuracy()
    # precision = BinaryPrecision()
    # recall = BinaryRecall()
    # specificity = BinarySpecificity()

    # mean_jaccard = MeanMetric()
    # mean_dice = MeanMetric()
    # mean_accuracy = MeanMetric()
    # mean_precision = MeanMetric()
    # mean_recall = MeanMetric()
    # mean_specificity = MeanMetric()

    def __init__(self):
        self.jaccard = BinaryJaccardIndex()
        self.dice =  Dice()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.specificity = BinarySpecificity()

        self.mean_jaccard = MeanMetric()
        self.mean_dice = MeanMetric()
        self.mean_accuracy = MeanMetric()
        self.mean_precision = MeanMetric()
        self.mean_recall = MeanMetric()
        self.mean_specificity = MeanMetric()
        self.metrics = [(self.jaccard, self.mean_jaccard), (self.dice, self.mean_dice), (self.accuracy, self.mean_accuracy), (self.precision, self.mean_precision), (self.recall, self.mean_recall), (self.specificity, self.mean_specificity)]


    def reset_metrics(self):
        for metric, _ in self.metrics:
            metric.reset()

    def update_metrics(self, pred, target):
        for metric, _ in self.metrics:
            metric.update(pred, target)
    
    def update_and_compute(self, pred, target):
        
        self.reset_metrics()
        self.update_metrics(pred, target)
        values = []

        for metric, mean in self.metrics:
            val = metric.compute()
            mean.update(val)
            mean = mean.compute()
            values.append((val, mean))
        return values
 
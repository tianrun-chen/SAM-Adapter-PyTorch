from torchmetrics.classification import BinaryJaccardIndex # IoU
from torchmetrics.classification import Dice
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall 
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAUROC
# Wrapper Class around BinaryJaccardIndex with it mean value
class JaccardIndex:
    def __init__(self, device='cuda'):
        self.jaccard = BinaryJaccardIndex().to(device)
        self.mean_jaccard = MeanMetric().to(device)

    def reset(self):
        self.jaccard.reset()
        self.mean_jaccard.reset()

    def reset_current(self):
        self.jaccard.reset()
    
    def update(self, pred, target):
        self.jaccard.update(pred, target)
        self.mean_jaccard.update(self.jaccard.compute())

    def compute(self):
        return self.jaccard.compute(), self.mean_jaccard.compute()
    
 # Wrapper Class around Dice with its mean value
class DiceCoefficient:
    def __init__(self, device='cuda'):
        self.dice = Dice().to(device)
        self.mean_dice = MeanMetric().to(device)
        self.mean_overall = -1

    def reset(self):
        self.dice.reset()
        self.mean_dice.reset()
    
    def reset_current(self):
        self.dice.reset()

    def update(self, pred, target):
        self.dice.update(pred, target)
        self.mean_dice.update(self.dice.compute())

    def compute(self):
        return self.dice.compute(), self.mean_dice.compute()

# Wrapper Class around Precision and its mean value

class Precision:
    def __init__(self, device='cuda'):
        self.precision = BinaryPrecision().to(device)
        self.mean_precision = MeanMetric().to(device)
    
    def reset(self):
        self.precision.reset()
        self.mean_precision.reset()
    
    def reset_current(self):
        self.precision.reset()
    
    def update(self, pred, target):
        self.precision.update(pred, target)
        self.mean_precision.update(self.precision.compute())
    
    def compute(self):
        return self.precision.compute(), self.mean_precision.compute()
    
# Wrapper Class around Recall and its mean value
class Recall:
    def __init__(self, device='cuda'):
        self.recall = BinaryRecall().to(device)
        self.mean_recall = MeanMetric().to(device)
    
    def reset(self):
        self.recall.reset()
        self.mean_recall.reset()
    
    def reset_current(self):
        self.recall.reset()
    
    def update(self, pred, target):
        self.recall.update(pred, target)
        self.mean_recall.update(self.recall.compute())
    
    def compute(self):
        return self.recall.compute(), self.mean_recall.compute()

# Wrapper Class around Accuracy and its mean value
class Accuracy:
    def __init__(self, device='cuda'):
        self.accuracy = BinaryAccuracy().to(device)
        self.mean_accuracy = MeanMetric().to(device)
    
    def reset(self):
        self.accuracy.reset()
        self.mean_accuracy.reset()
    
    def reset_current(self):
        self.accuracy.reset()
    
    def update(self, pred, target):
        self.accuracy.update(pred, target)
        self.mean_accuracy.update(self.accuracy.compute())
    
    def compute(self):
        return self.accuracy.compute(), self.mean_accuracy.compute()

# Wrapper Class around F1Score and its mean value
class F1Score:
    def __init__(self, device='cuda'):
        self.f1score = BinaryF1Score().to(device)
        self.mean_f1score = MeanMetric().to(device)
    
    def reset(self):
        self.f1score.reset()
        self.mean_f1score.reset()
    
    def reset_current(self):
        self.f1score.reset()
    
    def update(self, pred, target):
        self.f1score.update(pred, target)
        self.mean_f1score.update(self.f1score.compute())
    
    def compute(self):
        return self.f1score.compute(), self.mean_f1score.compute()
    
# Wrapper Class around aucroc and its mean value
class AUCROC:
    def __init__(self, device='cuda'):
        self.aucroc = BinaryAUROC().to(device)
        self.mean_aucroc = MeanMetric().to(device)
    
    def reset(self):
        self.aucroc.reset()
        self.mean_aucroc.reset()
    
    def reset_current(self):
        self.aucroc.reset()
    
    def update(self, pred, target):
        self.aucroc.update(pred, target)
        self.mean_aucroc.update(self.aucroc.compute())
    
    def compute(self):
        return self.aucroc.compute(), self.mean_aucroc.compute()
# Metrics Class with defined metric wrappers with ability to dynamically add metrics if needed
class Metrics:
    def __init__(self, metrics, device='cuda'):
        self.metrics = metrics
        self.metrics_dict = {}
        for metric in metrics:
            self.metrics_dict[metric] = eval(metric)(device=device)

    def reset(self):
        for metric in self.metrics:
            self.metrics_dict[metric].reset()
    
    def reset_current(self):
        for metric in self.metrics:
            self.metrics_dict[metric].reset_current()

    def update(self, pred, target):
        for metric in self.metrics:
            self.metrics_dict[metric].update(pred, target)

    def compute(self):
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = self.metrics_dict[metric].compute()
        return metrics
    
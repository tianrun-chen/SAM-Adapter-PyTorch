from torchmetrics.classification import BinaryJaccardIndex # IoU
from torchmetrics.classification import Dice
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall 
from torchmetrics.aggregation import MeanMetric

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
    
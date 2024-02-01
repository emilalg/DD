# Importing the 'base' class from the current package context. This is a generic class for metrics.
from . import base
# Importing functions from the 'functional' module within the current package, using 'F' as an alias.
from . import functional as F
# Importing the 'Activation' class from a 'modules' sub-package located one level up in the package hierarchy.
from ..base.modules import Activation

import torch.nn as nn
import torch

# IoU (Intersection over Union) is a metric for object detection and segmentation tasks.
# It computes the ratio of the area of overlap (intersection) between the predicted segmentation and the ground truth 
# to the area encompassed by both (union). IoU values range from 0 (no overlap) to 1 (perfect overlap).
# Higher IoU values indicate more accurate predictions.
class IoU(base.Metric):
    __name__ = 'iou_score'
    # The constructor of the IoU class which initializes the default parameters and can take additional keyword arguments.
    # 'eps' is a small value to prevent division by zero. 'threshold' is the cutoff for deciding positive class in predictions.
    # 'activation' applies a specific activation function to predictions, and 'ignore_channels' allows excluding 
    # specific channels in the computation for multi-channel data.
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
    # The forward method applies the activation to the predictions and computes the IoU score.
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

# The Fscore class is used for calculating the F1 score, which is the harmonic mean of precision and recall.
# The 'beta' parameter determines the weight of precision and recall in the harmonic mean, with beta=1 giving equal weight.
# It balances the trade-off between precision (correct positive predictions) and recall (detecting all positives).
# F1 score ranges from 0 (worst) to 1 (best), with higher values indicating better model performance.
class Fscore(base.Metric):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta # The beta parameter shapes the trade-off between precision and recall.
        self.threshold = threshold
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
    # The 'forward' method applies the activation function to predictions and calculates the F-score.
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

# The Accuracy class calculates the accuracy of the model's predictions, which is the proportion of correct predictions over the total number of cases.
# This metric is useful for balanced classification problems but can be misleading if the classes are imbalanced.
class Accuracy(base.Metric):

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr, y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

# Recall (Sensitivity/True Positive Rate) measures the ability of a model to identify all relevant cases (true positives).
# High recall means the model is good at detecting positive instances, but it doesn’t consider false positives.
# Recall ranges from 0 to 1:
# - 0 indicates no true positives are identified (worst scenario).
# - 1 indicates all true positives are identified (best scenario).
class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

# Precision measures the accuracy of positive predictions made by the model.
# It is the ratio of true positives to all predicted positives (true positives and false positives).
# High precision indicates that the model's positive predictions are likely to be correct.
# Precision ranges from 0 to 1:
# - 0 indicates no true positives in all positive predictions (worst scenario).
# - 1 indicates perfect precision where all positive predictions are true positives (best scenario).
class Precision(base.Metric):

    # The constructor of the Precision class which initializes the default parameters and can take additional keyword arguments.
    # 'eps' is a small value to prevent division by zero. 'threshold' is the cutoff for deciding positive class in predictions.
    # 'activation' applies a specific activation function to predictions, and 'ignore_channels' allows excluding 
    # specific channels in the computation for multi-channel data.
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr) # Applying the activation function to the predicted values.
        # Calling the precision function from the 'functional' module with the necessary parameters.
        return F.precision(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(base.Loss):
    def __init__(self, eps=1e-7, beta=1.0, gamma=2.5, activation=None, ignore_channels=None):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.gamma = gamma
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)

        C = y_pr.size(1)  # Number of classes
        loss = 0.0

        for c in range(C):
            if self.ignore_channels is not None and c in self.ignore_channels:
                continue

            y_true_c = y_gt[:, c, ...]
            y_pred_c = y_pr[:, c, ...]

            tp = torch.sum(y_pred_c * y_true_c, dim=[0, 1, 2])
            fp = torch.sum(y_pred_c * (1 - y_true_c), dim=[0, 1, 2])
            fn = torch.sum((1 - y_pred_c) * y_true_c, dim=[0, 1, 2])

            fp = fp * ((1 - y_true_c) ** self.gamma)
            fn = fn * ((1 - y_pred_c) ** self.gamma)

            numerator = (1 + self.beta**2) * tp + self.eps
            denominator = (1 + self.beta**2) * tp + self.beta**2 * fn + fp + self.eps

            class_loss = 1 - numerator / denominator
            loss += class_loss

        final_loss = loss.mean()  # Reducing the loss to a scalar
        return final_loss
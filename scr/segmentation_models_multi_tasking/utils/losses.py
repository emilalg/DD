import torch.nn as nn
import torch
from . import base
from . import functional as F
from .base import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DSCPlusPlusLoss(base.Loss):
    def __init__(self, eps=1e-5, beta=1.0, gamma=2.5, activation=None, ignore_channels=None):
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


class TverskyLoss(base.Loss):
    def __init__(self, alpha=0.3, eps=1.0, beta=0.7, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.tversky_score(
            y_pr,
            y_gt,
            alpha=self.alpha,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class FocalTverskyLoss(base.Loss):
    def __init__(
        self, alpha=0.3, eps=1.0, beta=0.7, gamma=0.75, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        tversky_loss = 1 - F.tversky_score(
            y_pr,
            y_gt,
            alpha=self.alpha,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

        return torch.pow(tversky_loss, self.gamma)



class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.binary_cross_entropy_with_logits(
            y_pr,
            y_gt,
            threshold=None,
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
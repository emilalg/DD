import torch.nn as nn
import torch
from . import base
from . import functional as F
from .base import Activation
from torch.autograd import Variable
from itertools import filterfalse


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

"""
Created on [04.02.2024]

FocalTverskyPlusPlusLoss implements a custom loss function class for segmentation models.
This loss function is designed to address class imbalance and enhance the model's focus on difficult-to-classify examples.
It combines the Tversky loss with a focal mechanism, adjusting the standard Tversky loss (a generalization of Dice loss)
by raising errors on hard examples to a power, thereby amplifying their contribution to the total loss.
Additionally, it incorporates the 'plus plus' component of the Dice loss Plus Plus (DSC++), providing a more refined approach
to model training that emphasizes learning from the structural complexity within the data.
The class supports ignoring specific channels during loss calculation and applies an activation function
to the model's predictions as part of the loss computation process.

"""
class FocalTverskyPlusPlusLoss(base.Loss):

    # The initializer method for the class.
    def __init__(self, alpha=0.3, eps=1e-7, beta=0.7, gamma=2.5, activation=None, ignore_channels=None):
        super().__init__()  # Call the initializer of the base class (base.Loss)
        self.eps = eps  # A small epsilon value to avoid division by zero
        self.alpha = alpha  # Alpha parameter of the Tversky index (controls the balance between false positives and false negatives)
        self.beta = beta  # Beta parameter of the Tversky index (controls the balance between false positives and false negatives)
        self.gamma_ftl = gamma  # Gamma parameter for the focal component (controls the focus on hard examples)
        self.activation = Activation(activation)  # Activation function to be applied to the predictions before computing the loss
        self.ignore_channels = ignore_channels  # Channels to ignore when computing the loss

    # The forward method is where the loss computation takes place.
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)  # Apply the activation function to the predictions
        
        C = y_pr.size(1)  # Number of channels/classes in the predictions
        loss = 0.0  # Initialize the loss to zero

        # Iterate over each channel/class
        for c in range(C):
            if self.ignore_channels is not None and c in self.ignore_channels:
                continue  # Skip the channels that are to be ignored

            y_true_c = y_gt[:, c, ...]  # Extract the ground truth for the current channel
            y_pred_c = y_pr[:, c, ...]  # Extract the predictions for the current channel

            # Calculate true positives, false positives, and false negatives
            tp = torch.sum(y_pred_c * y_true_c, dim=[0, 1, 2])  # True Positives
            fp = torch.sum(y_pred_c * (1 - y_true_c), dim=[0, 1, 2])  # False Positives
            fn = torch.sum((1 - y_pred_c) * y_true_c, dim=[0, 1, 2])  # False Negatives

            # Adjust the false positives and negatives with the gamma parameter, which is a modification from DSC++
            fp = fp * ((1 - y_true_c) ** self.gamma_ftl)
            fn = fn * ((1 - y_pred_c) ** self.gamma_ftl)

            # Calculate the Tversky index (a generalization of the Dice coefficient)
            numerator = (1 + self.beta ** 2) * tp + self.eps
            denominator = (1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.eps

            # Compute the class-specific loss
            class_loss = 1 - numerator / denominator
            # Apply the focal component by raising the loss to the power of the gamma parameter
            loss += torch.pow(class_loss, self.gamma_ftl)

        # Calculate the mean loss across all classes to get a single scalar value
        final_loss = loss.mean()
        return final_loss


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

#THESE ARE FOR THE COMBO LOSS FUNCTION
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
BETA = 0.5
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
e = eps=1e-7
"""
Created on [04.02.2024]

The ComboLoss class is a custom loss function that combines the effects of Dice loss and 
cross-entropy to create a robust loss for segmentation tasks. This combination helps to deal with 
class imbalance by leveraging the strengths of both losses: the Dice loss is good for handling 
imbalanced data, and the cross-entropy loss ensures that the model learns from all classes.

This loss function is particularly useful in tasks where the region of interest 
might be significantly smaller than the background, leading to an imbalance between the classes.

The forward method of this class computes the Combo loss given the model's predictions and 
the true labels. It flattens the tensors, calculates the intersection (or True Positives), 
and then computes the Dice coefficient. Additionally, it applies a weighted cross-entropy loss, 
which is modulated by the ALPHA and CE_RATIO parameters to balance the contribution of both Dice 
and cross-entropy components to the final loss value.

"""
class ComboLoss(base.Loss):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()
        # weight parameter for weighted cross-entropy loss
        # size_average to average the loss over the batch

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        # Calculate the Combo loss given the inputs and targets,
        # where 'inputs' are the predicted probabilities from the model
        # and 'targets' are the true labels.
        
        # Flatten the input and target tensors to ensure the calculation is done
        # on the same dimensions.
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate the True Positives (intersection) for the Dice score.
        intersection = (inputs * targets).sum()    
        # Calculate the Dice score, which is a measure of overlap between the
        # predicted segmentation and the ground truth. 
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Clamp the inputs to avoid log(0) which would result in NaN.
        # 'e' should be a small constant (e.g., 1e-7).
        inputs = torch.clamp(inputs, e, 1.0 - e)       
        # Calculate the weighted cross-entropy loss.
        out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
        # Take the mean of the weighted cross-entropy loss across all observations.
        weighted_ce = out.mean(-1)
        # Combine the weighted cross-entropy loss with the Dice loss.
        combo = (CE_RATIO * weighted_ce) + ((1 - CE_RATIO) * dice)
        
        # Return the combined loss.
        return combo
    
#UNDER CONSTRUCTION-----------------LOVAZ SOFTMAX LETS START DATING---------------
class LovaszSoftmaxLoss(base.Loss):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets):
        probas = F.softmax(inputs, dim=1) # B*C*H*W -> from logits to probabilities
        return lovasz_softmax(probas, targets, self.classes, self.per_image, self.ignore)

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

# --------------------------- HELPER FUNCTIONS FOR LOVAZ SOFTMAX ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
#-----------------END OF LOVAZ SOFTMAX THOU THIS COULD BE TRUE LOVE---------------

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
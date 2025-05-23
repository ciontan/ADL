import torch
import torch.nn as nn
import torch.nn.functional as F
# class FocalLoss(nn.Module):
#     """
#     The alpha parameter adjusts the weight for the minority class.
#     The gamma parameter adjusts how much to focus on hard examples (higher values will focus more on adifficult-to-classify samples).

#     """
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-BCE_loss)  # pt is the probability for each class
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = probs.clamp(min=1e-6, max=1 - 1e-6)  # avoid log(0)

        targets = targets.type_as(inputs)

        # focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
class DiceLoss(nn.Module):
    """
    Dice loss is a metric commonly used for imbalanced datasets, especially in segmentation tasks. It measures the overlap between the predicted and true classes. While it’s more often used in segmentation, it can be adapted for binary classification tasks.
 
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

#! default_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0])).to(device)
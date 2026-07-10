import torch
import torch.nn as nn
import torch.nn.functional as F

from src.framework.core.registry import LOSSES
from src.framework.losses.dice_loss import DiceLoss


@LOSSES.register("combined")
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=10.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

    def forward(self, pred_logits, target):
        device = pred_logits.device
        pos_weight = torch.tensor([self.pos_weight], device=device)
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
        dice = self.dice_loss(pred_logits, target)
        return self.bce_weight * bce + self.dice_weight * dice

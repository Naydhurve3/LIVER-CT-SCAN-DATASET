import torch
import torch.nn as nn
import torch.nn.functional as F

from src.framework.core.registry import LOSSES


@LOSSES.register("uwacl_v1")
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, beta=5.0, tau=0.1, dice_weight=0.5, bce_weight=0.5, pos_weight=10.0, smooth=1e-6):
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.smooth = smooth

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, pred_logits, target, uncertainty=None):
        batch_size = pred_logits.size(0)
        b, c, h, w = pred_logits.shape
        n = b * c * h * w
        device = pred_logits.device

        pos_weight = torch.tensor([self.pos_weight], device=device)
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight, reduction='none')

        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_per_sample = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice = dice_per_sample.view(b, c, 1, 1).expand_as(pred_logits)

        per_pixel_loss = self.bce_weight * bce + self.dice_weight * dice

        if uncertainty is not None:
            if uncertainty.device != device:
                uncertainty = uncertainty.to(device)
            weight_map = 1.0 + self.beta * (1.0 - torch.exp(-uncertainty / max(self.tau, 1e-8)))
            per_pixel_loss = per_pixel_loss * weight_map

        return per_pixel_loss.sum() / n

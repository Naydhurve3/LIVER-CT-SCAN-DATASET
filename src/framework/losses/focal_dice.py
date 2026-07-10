import torch
import torch.nn as nn
import torch.nn.functional as F

from src.framework.core.registry import LOSSES


@LOSSES.register("focal_dice")
class FocalDiceLoss(nn.Module):
    """Binary focal loss plus soft Dice for sparse tumor masks."""

    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0,
                 focal_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs)
        alpha = torch.where(
            target > 0.5,
            torch.as_tensor(self.focal_alpha, device=logits.device, dtype=logits.dtype),
            torch.as_tensor(1.0 - self.focal_alpha, device=logits.device, dtype=logits.dtype),
        )
        focal = (alpha * (1.0 - pt).pow(self.focal_gamma) * bce).mean()
        dims = (1, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        denominator = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) /
                      (denominator + self.smooth)).mean()
        return self.focal_weight * focal + self.dice_weight * dice

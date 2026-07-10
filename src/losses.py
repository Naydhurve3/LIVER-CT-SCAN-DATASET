import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting.

    gamma: how much to focus on hard examples (2.0 = standard)
    alpha: weight for the POSITIVE (tumor) class. Background gets 1 - alpha.
           alpha > 0.5 means tumor pixels contribute more to the loss.
           Default 0.75 means tumor gets 3x the weight of background.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-bce)
        # Per-class alpha: alpha for positive class, 1-alpha for background
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice Loss combined.

    Focal handles extreme class imbalance by down-weighting easy bg pixels.
    Dice directly penalizes all-background predictions (Dice=0 -> loss=1).
    Together they prevent background collapse better than either alone.
    """
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_logits, target):
        return (self.focal_weight * self.focal(pred_logits, target) +
                self.dice_weight * self.dice(pred_logits, target))


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=10.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, pred_logits, target):
        device = pred_logits.device
        pos_weight = self.pos_weight.to(device)
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pos_weight)
        dice = self.dice_loss(pred_logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, beta=5.0, tau=0.1, dice_weight=0.5, bce_weight=0.5, pos_weight=10.0, smooth=1e-6):
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight])
        self.smooth = smooth

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, pred_logits, target, uncertainty=None):
        batch_size = pred_logits.size(0)
        b, c, h, w = pred_logits.shape
        n = b * c * h * w
        device = pred_logits.device

        pos_weight = self.pos_weight.to(device)
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


if __name__ == "__main__":
    print(f"=== {__file__} ===")
    pred = torch.randn(4, 1, 64, 64)
    target = (torch.rand(4, 1, 64, 64) > 0.5).float()
    dl = DiceLoss()
    cl = CombinedLoss()
    fl = FocalLoss()
    uwl = UncertaintyWeightedLoss()
    print(f"  DiceLoss: {dl(pred, target):.4f}")
    print(f"  CombinedLoss: {cl(pred, target):.4f}")
    print(f"  FocalLoss: {fl(pred, target):.4f}")
    print(f"  UncertaintyWeightedLoss: {uwl(pred, target):.4f}")
    print("  OK")

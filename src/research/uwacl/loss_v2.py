import torch
import torch.nn as nn
import torch.nn.functional as F

from src.framework.core.registry import LOSSES


@LOSSES.register("uwacl_v2_multi")
class UWACLv2MultiScale(nn.Module):
    def __init__(self, beta=5.0, tau=0.1, tau_min=0.01,
                 dice_weight=0.5, bce_weight=0.5, pos_weight=10.0,
                 smooth=1e-6, edge_weight=0.1,
                 scale_weights=(0.4, 0.35, 0.25)):
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.tau_min = tau_min
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.smooth = smooth
        self.edge_weight = edge_weight
        self.scale_weights = scale_weights
        self._current_tau = tau
        self._current_epoch = 0

    def set_tau(self, tau):
        self._current_tau = tau

    def set_epoch(self, epoch, max_epochs=50):
        self._current_epoch = epoch
        decay = epoch / max(max_epochs, 1)
        self._current_tau = self.tau * (1.0 - decay) + self.tau_min * decay

    def _multi_scale_uncertainty(self, uncertainty):
        if uncertainty is None:
            return None
        fine = uncertainty
        medium = F.avg_pool2d(uncertainty, kernel_size=3, stride=1, padding=1)
        coarse = F.avg_pool2d(uncertainty, kernel_size=7, stride=1, padding=3)
        w0, w1, w2 = self.scale_weights
        return w0 * fine + w1 * medium + w2 * coarse

    def _edge_loss(self, pred, target):
        pred_edge = pred - F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        target_edge = target - F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
        return F.mse_loss(pred_edge, target_edge)

    def forward(self, pred_logits, target, uncertainty=None):
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
            u_multi = self._multi_scale_uncertainty(uncertainty)
            if u_multi.device != device:
                u_multi = u_multi.to(device)
            weight_map = 1.0 + self.beta * (1.0 - torch.exp(-u_multi / max(self._current_tau, 1e-8)))
            per_pixel_loss = per_pixel_loss * weight_map

        edge = self._edge_loss(pred, target) if self.edge_weight > 0 else 0.0
        return per_pixel_loss.sum() / n + self.edge_weight * edge

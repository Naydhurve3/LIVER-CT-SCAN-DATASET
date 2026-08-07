import torch
import torch.nn as nn
import torch.nn.functional as F

from src.framework.core.registry import LOSSES


@LOSSES.register("stability_bounded_recall")
class StabilityBoundedRecallLoss(nn.Module):
    """Recall-aware Focal-Tversky-style loss with a FIXED upper cap.

    Motivation
    ----------
    The prior unbounded Focal-Tversky ablation collapsed to empty masks
    (zero predicted tumour pixels), so a capped analogue is needed to make the
    false-negative emphasis safe. This class keeps a recall-heavy Tversky term
    but bounds its effective false-negative weight by a fixed ``fn_cap`` and
    blends in a stable Focal-Dice term so the loss cannot drive the model into
    a degenerate empty foreground.

    Design
    ------
    ``loss = w_dice * FocalDice + w_tv * FocalTversky`` where the Tversky index
    uses a capped false-negative penalty:

    .. math::
        TV = (TP + s) / (TP + s + a*FP + min(b, cap)*FN)

    All arithmetic is carried out in float32 to avoid fp16 fractional-power NaNs.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 1.30,
        gamma: float = 0.70,
        cap: float = 2.00,
        smooth: float = 1e-2,
        dice_weight: float = 0.50,
        tversky_weight: float = 0.50,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0 or cap < 0:
            raise ValueError("alpha, beta and cap must be non-negative.")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cap = cap
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

    def _effective_beta(self) -> float:
        # A FIXED upper bound on the false-negative weight is the C3 safeguard.
        return min(self.beta, self.cap)

    def _tversky(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, probs.ndim))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        eff_beta = self._effective_beta()
        score = (tp + self.smooth) / (
            tp + self.alpha * fp + eff_beta * fn + self.smooth
        )
        score = score.clamp(min=0.0, max=1.0)
        error = (1.0 - score).clamp(min=0.0, max=1.0)
        return torch.pow(error.clamp(min=1e-6), self.gamma).mean()

    def _dice(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        target = target.float()
        return (
            self.dice_weight * self._dice(probs, target)
            + self.tversky_weight * self._tversky(probs, target)
        )


_REGISTERED = "stability_bounded_recall"
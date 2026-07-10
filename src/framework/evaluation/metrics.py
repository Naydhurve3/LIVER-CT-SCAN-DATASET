import torch

from src.framework.core.registry import METRICS


def dice_coefficient(pred_binary, target, smooth=1e-6):
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred_binary, target, smooth=1e-6):
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def per_slice_metrics(pred_binary, target):
    batch_size = pred_binary.size(0)
    dices, ious = [], []
    for i in range(batch_size):
        p = pred_binary[i:i+1]
        t = target[i:i+1]
        dices.append(dice_coefficient(p, t).item())
        ious.append(iou_score(p, t).item())
    return torch.tensor(dices), torch.tensor(ious)


def ensemble_uncertainty(preds_list):
    stacked = torch.stack(preds_list, dim=0)
    mean_pred = stacked.mean(dim=0)
    variance = stacked.var(dim=0)
    return mean_pred, variance


def calibration_error(probs, targets, n_bins=10):
    confidences = probs.view(-1)
    accuracies = (probs > 0.5).float().view(-1).eq(targets.view(-1)).float()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / confidences.numel()) * abs(bin_acc - bin_conf)
    return ece


@METRICS.register("dice")
class Dice:
    def __call__(self, pred_binary, target):
        return dice_coefficient(pred_binary, target).item()


@METRICS.register("iou")
class IoU:
    def __call__(self, pred_binary, target):
        return iou_score(pred_binary, target).item()


@METRICS.register("ece")
class ECE:
    def __call__(self, probs, targets):
        return calibration_error(probs, targets).item()

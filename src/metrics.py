import torch
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure


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


def tumor_only_dice(pred_binary, target, smooth=1e-6):
    tumor_mask = target.sum(dim=(1, 2, 3)) > 0.5  # (N,) bool
    if tumor_mask.sum() < 0.5:
        return torch.tensor(0.0, device=pred_binary.device)
    return dice_coefficient(pred_binary[tumor_mask], target[tumor_mask], smooth)


def tumor_only_iou(pred_binary, target, smooth=1e-6):
    tumor_mask = target.sum(dim=(1, 2, 3)) > 0.5
    if tumor_mask.sum() < 0.5:
        return torch.tensor(0.0, device=pred_binary.device)
    return iou_score(pred_binary[tumor_mask], target[tumor_mask], smooth)


def predicted_foreground_fraction(pred_binary):
    """Fraction of all pixels predicted as foreground (tumor)."""
    return pred_binary.sum().item() / max(pred_binary.numel(), 1)


def slice_level_auroc(pred_probs, target, n_thresh=1000):
    """Slice-level AUROC: can the model distinguish tumor slices from bg slices?
    
    Computes mean probability per slice, then ROC across slices.
    Sklearn is used for roc_auc_score; returns NaN if only one class present.
    """
    from sklearn.metrics import roc_auc_score
    # Mean probability per slice -> (N,) 
    slice_preds = pred_probs.mean(dim=(1, 2, 3)).cpu().numpy()
    slice_target = (target.sum(dim=(1, 2, 3)) > 0.5).cpu().numpy().astype(int)
    if len(set(slice_target)) < 2:
        return float('nan')
    return float(roc_auc_score(slice_target, slice_preds))


def slice_level_auprc(pred_probs, target):
    """Slice-level AUPRC (average precision). More informative than AUROC for imbalance."""
    from sklearn.metrics import average_precision_score
    slice_preds = pred_probs.mean(dim=(1, 2, 3)).cpu().numpy()
    slice_target = (target.sum(dim=(1, 2, 3)) > 0.5).cpu().numpy().astype(int)
    if slice_target.sum() < 1:
        return float('nan')
    return float(average_precision_score(slice_target, slice_preds))


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


def precision_recall(pred_binary, target, smooth=1e-6):
    tp = (pred_binary * target).sum(dim=(1, 2, 3))
    fp = (pred_binary * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_binary) * target).sum(dim=(1, 2, 3))
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision.mean(), recall.mean()


def sensitivity_specificity(pred_binary, target, smooth=1e-6):
    tp = (pred_binary * target).sum(dim=(1, 2, 3))
    fn = ((1 - pred_binary) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred_binary) * (1 - target)).sum(dim=(1, 2, 3))
    fp = (pred_binary * (1 - target)).sum(dim=(1, 2, 3))
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    return sensitivity.mean(), specificity.mean()


def _boundary_distance_map(mask_np):
    dt_in = distance_transform_edt(1 - mask_np)
    dt_out = distance_transform_edt(mask_np)
    return dt_in, dt_out


def _surface_voxels(mask_np):
    mask_bool = mask_np.astype(bool)
    struct = generate_binary_structure(mask_bool.ndim, 1)
    eroded = binary_erosion(mask_bool, struct)
    return mask_bool & (~eroded)


def _surface_dists(dt, surf_mask):
    coords = np.argwhere(surf_mask)
    if len(coords) == 0:
        return np.array([])
    return dt[tuple(coords.T)]


def hd95(pred_np, target_np, spacing=(1.0, 1.0, 1.0)):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float(max(pred_np.sum(), target_np.sum()))
    surf_pred = _surface_voxels(pred_np)
    surf_target = _surface_voxels(target_np)
    dt_pred, _ = _boundary_distance_map(pred_np)
    dt_target, _ = _boundary_distance_map(target_np)
    max_spacing = max(spacing)
    dists_fwd = _surface_dists(dt_target, surf_pred) * max_spacing
    dists_rev = _surface_dists(dt_pred, surf_target) * max_spacing
    all_dists = np.concatenate([dists_fwd, dists_rev])
    if len(all_dists) == 0:
        return 0.0
    return float(np.percentile(all_dists, 95))


def asd(pred_np, target_np, spacing=(1.0, 1.0, 1.0)):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float(max(pred_np.sum(), target_np.sum()))
    surf_pred = _surface_voxels(pred_np)
    surf_target = _surface_voxels(target_np)
    dt_pred, _ = _boundary_distance_map(pred_np)
    dt_target, _ = _boundary_distance_map(target_np)
    max_spacing = max(spacing)
    dists_fwd = _surface_dists(dt_target, surf_pred) * max_spacing
    dists_rev = _surface_dists(dt_pred, surf_target) * max_spacing
    all_dists = np.concatenate([dists_fwd, dists_rev])
    if len(all_dists) == 0:
        return 0.0
    return float(all_dists.mean())


def nsd(pred_np, target_np, spacing=(1.0, 1.0, 1.0), tau=2.0):
    if pred_np.sum() == 0 and target_np.sum() == 0:
        return 1.0
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return 0.0
    surf_pred = _surface_voxels(pred_np)
    surf_target = _surface_voxels(target_np)
    dt_pred, _ = _boundary_distance_map(pred_np)
    dt_target, _ = _boundary_distance_map(target_np)
    max_spacing = max(spacing)
    dists_fwd = _surface_dists(dt_pred, surf_target) * max_spacing
    dists_rev = _surface_dists(dt_target, surf_pred) * max_spacing
    numerator = int((dists_fwd <= tau).sum()) + int((dists_rev <= tau).sum())
    denominator = int(surf_target.sum()) + int(surf_pred.sum())
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def compute_all_metrics(pred_binary, target, probs=None, spacing=(1.0, 1.0, 1.0)):
    d = dice_coefficient(pred_binary, target).item()
    i = iou_score(pred_binary, target).item()
    td = tumor_only_dice(pred_binary, target).item()
    ti = tumor_only_iou(pred_binary, target).item()
    pr, re = precision_recall(pred_binary, target)
    se, sp = sensitivity_specificity(pred_binary, target)
    fg_frac = predicted_foreground_fraction(pred_binary)
    metrics = {
        'dice': d,
        'iou': i,
        'tumor_only_dice': td,
        'tumor_only_iou': ti,
        'precision': pr.item(),
        'recall': re.item(),
        'sensitivity': se.item(),
        'specificity': sp.item(),
        'pred_fg_fraction': fg_frac,
    }
    if probs is not None:
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        metrics['ece'] = calibration_error(probs_flat, target_flat).item()
        metrics['auroc'] = slice_level_auroc(probs, target)
        metrics['auprc'] = slice_level_auprc(probs, target)
    return metrics


def compute_volume_metrics(all_preds, all_targets, volume_ids, all_probs=None, spacing=(1.0, 1.0, 1.0)):
    unique_vids = sorted(set(volume_ids))
    results = {}
    per_vol_dices, per_vol_ious = [], []
    per_vol_tumor_dices, per_vol_tumor_ious = [], []
    per_vol_hd95, per_vol_asd, per_vol_nsd = [], [], []
    for vid in unique_vids:
        mask = [i for i, v in enumerate(volume_ids) if v == vid]
        pred_vol = (torch.cat([all_preds[i] for i in mask], dim=0) > 0.5).float()
        target_vol = torch.cat([all_targets[i] for i in mask], dim=0)
        pred_np = pred_vol.cpu().numpy().squeeze()
        target_np = target_vol.cpu().numpy().squeeze()
        pred_np_3d = (pred_np > 0).astype(np.uint8)
        target_np_3d = (target_np > 0).astype(np.uint8)
        d = dice_coefficient(pred_vol, target_vol).item()
        i = iou_score(pred_vol, target_vol).item()
        td = tumor_only_dice(pred_vol, target_vol).item()
        ti = tumor_only_iou(pred_vol, target_vol).item()
        h = hd95(pred_np_3d, target_np_3d, spacing)
        a = asd(pred_np_3d, target_np_3d, spacing)
        n = nsd(pred_np_3d, target_np_3d, spacing)
        per_vol_dices.append(d)
        per_vol_ious.append(i)
        per_vol_tumor_dices.append(td)
        per_vol_tumor_ious.append(ti)
        per_vol_hd95.append(h)
        per_vol_asd.append(a)
        per_vol_nsd.append(n)
        results[f'volume_{vid}'] = {
            'dice': d, 'iou': i,
            'tumor_only_dice': td, 'tumor_only_iou': ti,
            'hd95': h, 'asd': a, 'nsd': n,
        }
    results['aggregate'] = {
        'dice_mean': float(np.mean(per_vol_dices)),
        'dice_std': float(np.std(per_vol_dices)),
        'iou_mean': float(np.mean(per_vol_ious)),
        'tumor_only_dice_mean': float(np.mean(per_vol_tumor_dices)),
        'tumor_only_dice_std': float(np.std(per_vol_tumor_dices)),
        'tumor_only_iou_mean': float(np.mean(per_vol_tumor_ious)),
        'hd95_mean': float(np.mean(per_vol_hd95)),
        'hd95_std': float(np.std(per_vol_hd95)),
        'asd_mean': float(np.mean(per_vol_asd)),
        'nsd_mean': float(np.mean(per_vol_nsd)),
    }
    return results


if __name__ == "__main__":
    print(f"=== {__file__} ===")
    pred = torch.rand(4, 1, 64, 64)
    target = (torch.rand(4, 1, 64, 64) > 0.5).float()
    d = dice_coefficient((pred > 0.5).float(), target)
    i = iou_score((pred > 0.5).float(), target)
    td = tumor_only_dice((pred > 0.5).float(), target)
    ti = tumor_only_iou((pred > 0.5).float(), target)
    print(f"  Dice: {d:.4f}, IoU: {i:.4f}")
    print(f"  Tumor-only Dice: {td:.4f}, IoU: {ti:.4f}")
    pr, re = precision_recall((pred > 0.5).float(), target)
    print(f"  Precision: {pr:.4f}, Recall: {re:.4f}")
    se, sp = sensitivity_specificity((pred > 0.5).float(), target)
    print(f"  Sensitivity: {se:.4f}, Specificity: {sp:.4f}")
    ece = calibration_error(pred, target)
    print(f"  ECE: {ece:.4f}")
    vol_results = compute_volume_metrics(
        [(pred[i:i+1] > 0.5).float() for i in range(4)],
        [target[i:i+1] for i in range(4)],
        [0, 0, 1, 1]
    )
    print(f"  Vol aggregate dice: {vol_results['aggregate']['dice_mean']:.4f}")
    print("  OK")

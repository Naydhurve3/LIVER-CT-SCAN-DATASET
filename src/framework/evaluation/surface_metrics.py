import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt


def _binary_mask(mask, class_id=1):
    if mask.ndim == 3:
        return (mask == class_id).astype(np.uint8)
    return (mask == class_id).astype(np.uint8)


def _surface_voxels(binary_mask):
    struct = generate_binary_structure(binary_mask.ndim, 1)
    eroded = binary_erosion(binary_mask, struct)
    return binary_mask & ~eroded


def _surface_points(binary_mask, spacing):
    surf = _surface_voxels(binary_mask)
    points = np.argwhere(surf)
    return points * np.array(spacing)


def hausdorff_distance_95(mask_pred, mask_gt, spacing=(1.0, 1.0, 1.0), class_id=1):
    pred_bin = _binary_mask(mask_pred, class_id)
    gt_bin = _binary_mask(mask_gt, class_id)
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return np.nan
    pred_pts = _surface_points(pred_bin, spacing)
    gt_pts = _surface_points(gt_bin, spacing)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d1 = tree_gt.query(pred_pts)[0]
    d2 = tree_pred.query(gt_pts)[0]
    dists = np.concatenate([d1, d2])
    return float(np.percentile(dists, 95))


def average_surface_distance(mask_pred, mask_gt, spacing=(1.0, 1.0, 1.0), class_id=1):
    pred_bin = _binary_mask(mask_pred, class_id)
    gt_bin = _binary_mask(mask_gt, class_id)
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return np.nan
    pred_pts = _surface_points(pred_bin, spacing)
    gt_pts = _surface_points(gt_bin, spacing)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d1 = tree_gt.query(pred_pts)[0]
    d2 = tree_pred.query(gt_pts)[0]
    all_dists = np.concatenate([d1, d2])
    return float(all_dists.mean())


def normalized_surface_dice(mask_pred, mask_gt, spacing=(1.0, 1.0, 1.0), tolerance=1.0, class_id=1):
    pred_bin = _binary_mask(mask_pred, class_id)
    gt_bin = _binary_mask(mask_gt, class_id)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0
    pred_surf = _surface_voxels(pred_bin)
    gt_surf = _surface_voxels(gt_bin)
    if pred_surf.sum() == 0 or gt_surf.sum() == 0:
        return np.nan
    dt_pred = distance_transform_edt(~pred_surf.astype(bool), sampling=spacing)
    dt_gt = distance_transform_edt(~gt_surf.astype(bool), sampling=spacing)
    pred_acceptable = gt_surf & (dt_pred <= tolerance)
    gt_acceptable = pred_surf & (dt_gt <= tolerance)
    numerator = pred_acceptable.sum() + gt_acceptable.sum()
    denominator = pred_surf.sum() + gt_surf.sum()
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def compute_all_surface_metrics(mask_pred, mask_gt, spacing=(1.0, 1.0, 1.0), tolerance=1.0, class_id=1):
    return {
        "hd95": hausdorff_distance_95(mask_pred, mask_gt, spacing, class_id),
        "asd": average_surface_distance(mask_pred, mask_gt, spacing, class_id),
        "nsd": normalized_surface_dice(mask_pred, mask_gt, spacing, tolerance, class_id),
    }

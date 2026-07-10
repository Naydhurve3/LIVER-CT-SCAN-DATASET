import numpy as np
from src.framework.evaluation.surface_metrics import (
    hausdorff_distance_95, average_surface_distance,
    normalized_surface_dice, compute_all_surface_metrics,
)


def test_hd95_perfect_overlap():
    mask = np.zeros((10, 32, 32), dtype=np.int32)
    mask[3:7, 10:20, 10:20] = 1
    hd = hausdorff_distance_95(mask, mask)
    assert hd == 0.0


def test_hd95_no_overlap():
    pred = np.zeros((10, 32, 32), dtype=np.int32)
    pred[3:7, 10:20, 10:20] = 1
    gt = np.zeros((10, 32, 32), dtype=np.int32)
    gt[3:7, 20:30, 20:30] = 1
    hd = hausdorff_distance_95(pred, gt)
    assert hd > 0


def test_hd95_empty_pred():
    pred = np.zeros((10, 32, 32), dtype=np.int32)
    gt = np.zeros((10, 32, 32), dtype=np.int32)
    gt[3:7, 10:20, 10:20] = 1
    hd = hausdorff_distance_95(pred, gt)
    assert np.isnan(hd)


def test_asd_perfect():
    mask = np.zeros((10, 32, 32), dtype=np.int32)
    mask[3:7, 10:20, 10:20] = 1
    asd = average_surface_distance(mask, mask)
    assert asd == 0.0


def test_asd_no_overlap():
    pred = np.zeros((10, 32, 32), dtype=np.int32)
    pred[3:7, 10:20, 10:20] = 1
    gt = np.zeros((10, 32, 32), dtype=np.int32)
    gt[3:7, 20:30, 20:30] = 1
    asd = average_surface_distance(pred, gt)
    assert asd > 0


def test_nsd_perfect():
    mask = np.zeros((10, 32, 32), dtype=np.int32)
    mask[3:7, 10:20, 10:20] = 1
    nsd = normalized_surface_dice(mask, mask)
    assert abs(nsd - 1.0) < 1e-6


def test_nsd_no_overlap():
    pred = np.zeros((10, 32, 32), dtype=np.int32)
    pred[3:7, 10:20, 10:20] = 1
    gt = np.zeros((10, 32, 32), dtype=np.int32)
    gt[3:7, 20:30, 20:30] = 1
    nsd = normalized_surface_dice(pred, gt)
    assert nsd >= 0 and nsd <= 1


def test_nsd_both_empty():
    pred = np.zeros((10, 32, 32), dtype=np.int32)
    gt = np.zeros((10, 32, 32), dtype=np.int32)
    nsd = normalized_surface_dice(pred, gt)
    assert nsd == 1.0


def test_compute_all():
    mask = np.zeros((10, 32, 32), dtype=np.int32)
    mask[3:7, 10:20, 10:20] = 1
    result = compute_all_surface_metrics(mask, mask)
    assert "hd95" in result
    assert "asd" in result
    assert "nsd" in result

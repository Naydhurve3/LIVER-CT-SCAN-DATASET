import torch
from src.framework.evaluation.metrics import dice_coefficient, iou_score, calibration_error


def test_dice_perfect():
    pred = torch.ones(2, 1, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    d = dice_coefficient(pred, target)
    assert abs(d.item() - 1.0) < 1e-4


def test_dice_no_overlap():
    pred = torch.zeros(2, 1, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    d = dice_coefficient(pred, target)
    assert abs(d.item()) < 1e-4


def test_dice_half_overlap():
    pred = torch.zeros(2, 1, 64, 64)
    pred[:, :, :, :32] = 1.0
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, :, 32:] = 1.0
    d = dice_coefficient(pred, target)
    assert abs(d.item()) < 1e-4


def test_iou_perfect():
    pred = torch.ones(2, 1, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    i = iou_score(pred, target)
    assert abs(i.item() - 1.0) < 1e-4


def test_ece_perfect_calibration():
    probs = torch.ones(1000, 1, 1, 1) * 0.9
    targets = torch.ones(1000, 1, 1, 1)
    ece = calibration_error(probs, targets, n_bins=10)
    assert abs(ece.item() - 0.1) < 0.01

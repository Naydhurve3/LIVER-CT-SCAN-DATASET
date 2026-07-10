import unittest
import torch
from src.metrics import (
    dice_coefficient, iou_score, per_slice_metrics,
    ensemble_uncertainty, calibration_error,
)


class TestDiceCoefficient(unittest.TestCase):
    def test_perfect_overlap(self):
        pred = torch.ones(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        dice = dice_coefficient(pred, target)
        self.assertAlmostEqual(dice.item(), 1.0, places=5)

    def test_no_overlap(self):
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        dice = dice_coefficient(pred, target)
        self.assertAlmostEqual(dice.item(), 0.0, places=5)

    def test_half_overlap(self):
        pred = torch.ones(2, 1, 32, 32)
        pred[:, :, :, 16:] = 0
        target = torch.ones(2, 1, 32, 32)
        dice = dice_coefficient(pred, target)
        self.assertAlmostEqual(dice.item(), 2 / 3, places=4)

    def test_single_element(self):
        pred = torch.ones(1, 1, 1, 1)
        target = torch.ones(1, 1, 1, 1)
        dice = dice_coefficient(pred, target)
        self.assertAlmostEqual(dice.item(), 1.0, places=5)


class TestIoUScore(unittest.TestCase):
    def test_perfect_overlap(self):
        pred = torch.ones(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        iou = iou_score(pred, target)
        self.assertAlmostEqual(iou.item(), 1.0, places=5)

    def test_no_overlap(self):
        pred = torch.zeros(2, 1, 32, 32)
        target = torch.ones(2, 1, 32, 32)
        iou = iou_score(pred, target)
        self.assertAlmostEqual(iou.item(), 0.0, places=5)

    def test_half_overlap(self):
        pred = torch.ones(2, 1, 32, 32)
        pred[:, :, :, 16:] = 0
        target = torch.ones(2, 1, 32, 32)
        iou = iou_score(pred, target)
        self.assertAlmostEqual(iou.item(), 0.5, places=4)


class TestPerSliceMetrics(unittest.TestCase):
    def test_per_slice_metrics_shape(self):
        pred = torch.rand(4, 1, 16, 16) > 0.5
        target = torch.rand(4, 1, 16, 16) > 0.5
        dices, ious = per_slice_metrics(pred.float(), target.float())
        self.assertEqual(dices.shape, (4,))
        self.assertEqual(ious.shape, (4,))

    def test_per_slice_bounds(self):
        pred = torch.ones(3, 1, 16, 16)
        target = torch.ones(3, 1, 16, 16)
        dices, ious = per_slice_metrics(pred, target)
        self.assertTrue((dices >= 0).all())
        self.assertTrue((dices <= 1).all())
        self.assertTrue((ious >= 0).all())
        self.assertTrue((ious <= 1).all())


class TestEnsembleUncertainty(unittest.TestCase):
    def test_identical_predictions(self):
        pred = torch.sigmoid(torch.randn(4, 1, 16, 16))
        mean, var = ensemble_uncertainty([pred, pred, pred])
        self.assertEqual(mean.shape, (4, 1, 16, 16))
        self.assertEqual(var.shape, (4, 1, 16, 16))
        self.assertAlmostEqual(var.sum().item(), 0.0, places=5)

    def test_diverse_predictions(self):
        p1 = torch.zeros(2, 1, 8, 8)
        p2 = torch.ones(2, 1, 8, 8)
        p3 = torch.full((2, 1, 8, 8), 0.5)
        mean, var = ensemble_uncertainty([p1, p2, p3])
        self.assertTrue(var.sum().item() > 0)


class TestCalibrationError(unittest.TestCase):
    def test_perfect_calibration(self):
        probs = torch.ones(100)
        targets = torch.ones(100)
        ece = calibration_error(probs, targets, n_bins=10)
        self.assertAlmostEqual(ece.item(), 0.0, places=5)

    def test_poor_calibration(self):
        probs = torch.sigmoid(torch.randn(1000))
        targets = (probs > 0.5).float()
        noise_mask = torch.rand(1000) > 0.9
        targets[noise_mask] = 1.0 - targets[noise_mask]
        ece = calibration_error(probs, targets)
        self.assertGreaterEqual(ece.item(), 0.0)
        self.assertLessEqual(ece.item(), 1.0)


if __name__ == "__main__":
    unittest.main()

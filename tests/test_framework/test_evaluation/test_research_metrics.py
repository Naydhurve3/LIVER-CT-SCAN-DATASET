import math

import numpy as np
import torch

from src.framework.evaluation.research_metrics import (
    StreamingProbabilityMetrics, confusion_counts, dice_from_counts,
    evaluate_research_model, positive_slice_dice, select_threshold,
)


class IdentityLogitModel(torch.nn.Module):
    def forward(self, x):
        return x


def test_empty_mask_policy():
    assert dice_from_counts(0, 0, 0) == 1.0
    assert dice_from_counts(0, 2, 0) == 0.0


def test_confusion_counts_hand_computed():
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [1, 0]])
    assert confusion_counts(pred, target) == (1, 1, 1, 1)


def test_positive_slice_dice_excludes_empty_slices():
    pred = torch.tensor([[[[0.0]]], [[[1.0]]]])
    target = torch.tensor([[[[0.0]]], [[[1.0]]]])
    assert positive_slice_dice(pred, target).item() == 1.0


def test_streaming_probability_metrics_perfect_ranking():
    metric = StreamingProbabilityMetrics(probability_bins=32, calibration_bins=5)
    metric.update(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    result = metric.compute()
    assert result["auprc"] == 1.0
    assert result["auroc"] == 1.0
    assert 0 <= result["ece"] <= 1


def test_threshold_selection_uses_positive_volume_dice_then_precision():
    batch = {
        "image": torch.tensor([[[[-2.0, 2.0]]]]),
        "mask": torch.tensor([[[[0.0, 1.0]]]]),
        "volume_id": torch.tensor([1]),
        "slice_id": torch.tensor([0]),
    }
    result = select_threshold(IdentityLogitModel(), [batch], torch.device("cpu"), [0.1, 0.5, 0.9])
    assert result["selected_threshold"] == 0.5


def test_evaluate_research_model_aggregation_and_empty_policy():
    batches = [
        {"image": torch.tensor([[[[-10.0]]]]), "mask": torch.zeros(1, 1, 1, 1),
         "volume_id": torch.tensor([0]), "slice_id": torch.tensor([0])},
        {"image": torch.tensor([[[[10.0]]]]), "mask": torch.ones(1, 1, 1, 1),
         "volume_id": torch.tensor([1]), "slice_id": torch.tensor([0])},
    ]
    result = evaluate_research_model(IdentityLogitModel(), batches, torch.device("cpu"), 0.5)
    assert result["aggregate"]["micro_dice"] == 1.0
    assert result["aggregate"]["positive_volume_dice"] == 1.0
    assert result["aggregate"]["macro_volume_dice"] == 1.0
    assert len(result["slices"]) == 2
    assert len(result["volumes"]) == 2

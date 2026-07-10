from tools.report import baseline_sanity, continuation_decision


def _metrics():
    return {
        "foreground_prevalence": 0.01, "auprc": 0.2,
        "positive_slice_dice": 0.4, "precision": 0.3, "recall": 0.7,
        "pred_fg_fraction": 0.012, "true_fg_fraction": 0.01,
        "positive_volume_dice": 0.5, "ece": 0.1, "hd95": 10.0,
    }


def test_baseline_sanity_passes_valid_metrics():
    assert baseline_sanity(_metrics())["passed"]


def test_continuation_gate_accepts_dice_gain():
    baseline = _metrics()
    candidate = dict(baseline, positive_volume_dice=0.53)
    assert continuation_decision(baseline, candidate)["continue_to_multiseed"]

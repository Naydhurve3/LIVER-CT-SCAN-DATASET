import numpy as np
from src.framework.evaluation.statistics import (
    bootstrap_ci, wilcoxon_signed_rank, bootstrapped_ci_metrics,
)


def test_bootstrap_ci_mean():
    scores = np.random.randn(50) + 0.5
    result = bootstrap_ci(scores, n_resamples=100, seed=42)
    assert "lower" in result
    assert "upper" in result
    assert result["lower"] < result["upper"]


def test_bootstrap_ci_median():
    scores = np.random.randn(50) + 0.5
    result = bootstrap_ci(scores, n_resamples=100, statistic="median", seed=42)
    assert result["lower"] < result["upper"]


def test_bootstrap_ci_single_sample():
    result = bootstrap_ci([1.0], n_resamples=10)
    assert abs(result["mean"] - 1.0) < 1e-6


def test_wilcoxon_significant():
    a = np.random.randn(30)
    b = a + 0.5
    result = wilcoxon_signed_rank(a, b)
    assert "p_value" in result
    assert "statistic" in result


def test_wilcoxon_identical():
    a = np.ones(10)
    result = wilcoxon_signed_rank(a, a)
    assert result["p_value"] == 1.0


def test_wilcoxon_too_few():
    try:
        wilcoxon_signed_rank([1.0], [2.0])
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_bootstrapped_ci_metrics():
    metrics = {
        "dice": [0.85, 0.87, 0.83, 0.86, 0.84],
        "iou": [0.74, 0.77, 0.72, 0.75, 0.73],
    }
    results = bootstrapped_ci_metrics(metrics, n_resamples=100)
    assert "dice" in results
    assert "iou" in results

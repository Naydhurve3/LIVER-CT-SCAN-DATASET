import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(scores, n_resamples=1000, ci=0.95, statistic="mean", seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot_stats = []
    for _ in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        if statistic == "mean":
            boot_stats.append(np.mean(sample))
        elif statistic == "median":
            boot_stats.append(np.median(sample))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    alpha = (1.0 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return {
        "lower": lower,
        "upper": upper,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)),
        "n_resamples": n_resamples,
        "ci": ci,
        "statistic": statistic,
    }


def wilcoxon_signed_rank(scores_a, scores_b):
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length")
    if len(scores_a) < 2:
        raise ValueError("Need at least 2 paired samples")
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    diff = scores_a - scores_b
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    non_zero = diff[diff != 0]
    if len(non_zero) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    stat, p = wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "mean_diff": float(np.mean(diff)),
    }


def bootstrapped_ci_metrics(metrics_dict, n_resamples=1000, seed=42):
    results = {}
    for name, values in metrics_dict.items():
        if len(values) < 2:
            results[name] = {"error": "Insufficient samples"}
            continue
        results[name] = bootstrap_ci(
            np.array(values), n_resamples=n_resamples, seed=seed
        )
    return results

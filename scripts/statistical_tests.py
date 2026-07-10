"""S15: Statistical significance tests (paired bootstrap) comparing models."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm
from src.utils import setup_logging, logger


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000, metric="dice"):
    """Paired bootstrap: H0: mean(A) == mean(B). Returns (obs_diff, p_value, sig)."""
    scores_a, scores_b = np.array(scores_a), np.array(scores_b)
    n = len(scores_a)
    obs_diff = float(scores_a.mean() - scores_b.mean())
    combined = np.concatenate([scores_a, scores_b])
    grand_mean = combined.mean()
    centered_a = scores_a - scores_a.mean() + grand_mean
    centered_b = scores_b - scores_b.mean() + grand_mean

    count = 0
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        d = centered_a[idx].mean() - centered_b[idx].mean()
        if abs(d) >= abs(obs_diff):
            count += 1
    p = count / n_bootstrap
    return obs_diff, round(p, 6), p < 0.05


def run_all_tests():
    dice_path = Path(__file__).resolve().parent.parent / "results" / "per_sample_dice.json"
    if not dice_path.exists():
        logger.error("Run ensemble_evaluation.py first to generate per_sample_dice.json")
        return

    per_sample = json.loads(dice_path.read_text())
    comparisons = [
        ("best_model", "ensemble"),
        ("best_model", "member_0"),
        ("best_model", "member_1"),
        ("best_model", "member_2"),
        ("ensemble", "member_0"),
        ("ensemble", "member_1"),
        ("ensemble", "member_2"),
        ("member_0", "member_1"),
        ("member_1", "member_2"),
        ("member_0", "member_2"),
    ]

    all_tests = {}
    for name_a, name_b in comparisons:
        if name_a not in per_sample or name_b not in per_sample:
            continue
        key = f"{name_a}_vs_{name_b}"
        scores_a = per_sample[name_a]
        scores_b = per_sample[name_b]
        if len(scores_a) == 0 or len(scores_b) == 0:
            continue
        obs_diff, p, sig = paired_bootstrap_test(scores_a, scores_b)
        all_tests[key] = {
            "model_a": name_a, "model_b": name_b,
            "mean_a": round(np.mean(scores_a), 4),
            "mean_b": round(np.mean(scores_b), 4),
            "obs_diff": round(obs_diff, 4),
            "p_value": p,
            "significant_005": sig,
        }
        sig_str = "***" if sig else "n.s."
        print(f"  {name_a} vs {name_b}: diff={obs_diff:.4f}, p={p:.6f} {sig_str}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "statistical_tests.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_tests, indent=2))
    logger.info(f"Statistical tests saved to {out_path}")
    return all_tests


if __name__ == "__main__":
    setup_logging()
    run_all_tests()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.framework.evaluation.checkpoints import write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Generate research-validation figures and decision report")
    parser.add_argument("--evaluation", action="append", required=True,
                        help="Path to evaluation.json; pass once per experiment")
    parser.add_argument("--output-dir", default="experiments/research_validation/report")
    return parser.parse_args()


def baseline_sanity(metrics):
    prevalence = metrics["foreground_prevalence"]
    ratio = metrics["pred_fg_fraction"] / max(metrics["true_fg_fraction"], 1e-12)
    checks = {
        "auprc_2x_prevalence": metrics["auprc"] >= 2 * prevalence,
        "positive_slice_dice_gt_0_10": metrics["positive_slice_dice"] > 0.10,
        "precision_gt_0_10": metrics["precision"] > 0.10,
        "recall_non_degenerate": 0.01 < metrics["recall"] < 0.99,
        "foreground_fraction_in_range": 0.25 <= ratio <= 4.0,
    }
    return {"passed": all(checks.values()), "checks": checks, "pred_true_fg_ratio": ratio}


def continuation_decision(baseline, candidate):
    dice_gain = candidate["positive_volume_dice"] - baseline["positive_volume_dice"]
    dice_loss = baseline["positive_volume_dice"] - candidate["positive_volume_dice"]
    ece_gain = baseline["ece"] - candidate["ece"]
    hd95_gain = ((baseline["hd95"] - candidate["hd95"]) / baseline["hd95"]
                 if baseline["hd95"] and baseline["hd95"] == baseline["hd95"] else 0.0)
    passed = dice_gain >= 0.02 or (dice_loss <= 0.01 and (ece_gain >= 0.02 or hd95_gain >= 0.10))
    return {
        "continue_to_multiseed": passed, "positive_volume_dice_gain": dice_gain,
        "ece_reduction": ece_gain, "hd95_relative_reduction": hd95_gain,
    }


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluations = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.evaluation]
    if any(item.get("split") != "test" for item in evaluations):
        raise ValueError("Decision report requires held-out test evaluations")
    baseline = evaluations[0]
    sanity = baseline_sanity(baseline["aggregate"])
    decisions = {
        item["experiment"]: continuation_decision(baseline["aggregate"], item["aggregate"])
        for item in evaluations[1:]
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for item in evaluations:
        plot = item["plot_data"]
        ax.plot(plot["pr_recall"], plot["pr_precision"], label=item["experiment"])
    ax.set(xlabel="Recall", ylabel="Precision", title="Pixel Precision-Recall Curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for item in evaluations:
        plot = item["plot_data"]
        ax.plot(plot["calibration_confidence"], plot["calibration_observed"], marker="o",
                label=item["experiment"])
    ax.set(xlabel="Mean predicted probability", ylabel="Observed tumor fraction",
           title="Reliability Diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability_diagram.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Research Validation Decision Report", "",
        "> Single-seed model-selection evidence only; not a significance or publication claim.", "",
        "## Evidence basis", "",
        "Dataset geometry, imbalance, and split limitations follow `Practice/` and `docs/DATA_REFERENCE.md`.",
        "The task is tumor/lesion-mask segmentation using already-windowed PNG inputs.", "",
        "## Held-out results", "",
        "| Experiment | Micro Dice | Positive-slice Dice | Positive-volume Dice | AUPRC | Precision | Recall | ECE | HD95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evaluations:
        m = item["aggregate"]
        lines.append(
            f"| {item['experiment']} | {m['micro_dice']:.4f} | {m['positive_slice_dice']:.4f} | "
            f"{m['positive_volume_dice']:.4f} | {m['auprc']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['ece']:.4f} | {m['hd95']:.4f} |"
        )
    lines.extend(["", "## Gates", "", f"Baseline sanity gate: **{'PASS' if sanity['passed'] else 'FAIL'}**", ""])
    for name, passed in sanity["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(["", "## Research continuation", ""])
    if not decisions:
        lines.append("No research candidate was supplied; continuation decision is pending.")
    for name, decision in decisions.items():
        state = "GO" if decision["continue_to_multiseed"] else "NO-GO"
        lines.append(
            f"- **{name}: {state}** — positive-volume Dice change "
            f"{decision['positive_volume_dice_gain']:+.4f}, ECE reduction "
            f"{decision['ece_reduction']:+.4f}, HD95 reduction "
            f"{decision['hd95_relative_reduction']:+.1%}."
        )
    if not sanity["passed"]:
        lines.extend(["", "Stop architecture expansion and diagnose label, sampler, loss, and threshold behavior."])
    lines.extend(["", "## Known limitation", "",
                  "The sequential test split has materially higher tumor burden than train/validation. "
                  "Future publication work requires multiple seeds, volume-wise confidence intervals, and cross-validation."])
    (out / "RESEARCH_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "decision.json", {"baseline_sanity": sanity, "continuation": decisions})
    print(f"report={out / 'RESEARCH_DECISION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

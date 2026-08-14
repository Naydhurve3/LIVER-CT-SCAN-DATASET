"""Build a stable Focal-Dice plus Focal-Tversky loss ablation notebook."""

from copy import deepcopy
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "recall_aware_focal_tversky_loss_ablation.ipynb"
OUTPUT = ROOT / "Practice" / "stabilized_composite_loss_ablation.ipynb"

notebook = deepcopy(nbformat.read(SOURCE, as_version=4))

notebook.cells[0].source = """# Stabilized Composite Loss Ablation

This experiment fixes the pure Focal-Tversky empty-mask collapse while retaining
a controlled recall signal. It uses:

- **75% proven Focal-Dice baseline loss**
- **25% numerically safe Focal-Tversky loss**

Only the loss changes. Dataset, split, preprocessing, model, augmentation,
sampler, optimizer, schedule and validation remain frozen. The test split is
locked.
"""

notebook.cells[1].source = """## tl;dr

Pure Focal-Tversky failed at every epoch: zero predicted tumor pixels, zero
positive-slice recall and 100% positive slices predicted empty.

This fresh 10-epoch experiment restores the stable Focal-Dice objective as the
dominant component and adds a bounded 25% recall-aware term. Synthetic empty,
tiny-lesion and larger-lesion gradient checks must pass before training starts.
"""

notebook.cells[2].source = """## Context & Methods

### Key assumptions

- The previous null prediction is an objective-collapse failure, not a threshold
  problem; thresholds 0.30–0.90 were identically empty.
- The original Focal-Dice baseline is the trusted optimization anchor.
- Focal-Tversky remains useful only as a minority auxiliary term.
- A fresh initialization is required; the collapsed checkpoint must not resume.

### Single intervention

`Composite = 0.75 × FocalDice + 0.25 × FocalTversky`

Tversky uses `alpha=0.40`, `beta=0.60`, `gamma=0.75`, float32 accumulation and
bounded fractional-power inputs.
"""

setup = notebook.cells[3].source
setup = setup.replace(
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "recall_aware_loss_outputs"',
    'PURE_TVERSKY_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "recall_aware_loss_outputs"\n'
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "stabilized_composite_loss_outputs"',
)
setup = setup.replace(
    'LOSS_STRATEGY = "focal_tversky_a040_b060_g075"',
    'LOSS_STRATEGY = "focal_dice075_plus_focal_tversky025"\n'
    "FOCAL_DICE_WEIGHT = 0.75\n"
    "FOCAL_TVERSKY_WEIGHT = 0.25",
)
notebook.cells[3].source = setup

loss_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code" and "class FocalTverskyLoss" in cell.source
)
old_loss_cell = notebook.cells[loss_index].source
model_suffix = old_loss_cell[old_loss_cell.index(
    "from src.framework.models.mobilenetv2_unet import MobileNetV2UNet"
):]
notebook.cells[loss_index].source = """import torch.nn as nn
from src.framework.losses.focal_dice import FocalDiceLoss


class FocalTverskyLoss(nn.Module):
    \"\"\"Numerically safe auxiliary recall-aware overlap loss.\"\"\"
    def __init__(self, alpha=0.40, beta=0.60, gamma=0.75, smooth=1.0):
        super().__init__()
        if not math.isclose(alpha + beta, 1.0, abs_tol=1e-8):
            raise ValueError("Tversky alpha and beta must sum to 1.")
        self.alpha, self.beta = alpha, beta
        self.gamma, self.smooth = gamma, smooth

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits.float())
        targets = targets.float()
        dimensions = tuple(range(1, logits.ndim))
        true_positive = (probabilities * targets).sum(dim=dimensions)
        false_positive = (probabilities * (1.0 - targets)).sum(dim=dimensions)
        false_negative = ((1.0 - probabilities) * targets).sum(dim=dimensions)
        score = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive
            + self.beta * false_negative + self.smooth
        )
        # gamma < 1 has an infinite derivative at exactly zero. A small floor
        # prevents 0 ** gamma from producing NaN gradients under AMP.
        focal_error = (1.0 - score.clamp(0.0, 1.0)).clamp(1e-6, 1.0)
        return torch.pow(focal_error, self.gamma).mean()


class StableCompositeLoss(nn.Module):
    \"\"\"Stable baseline objective plus a bounded recall-aware auxiliary term.\"\"\"
    def __init__(
        self, focal_dice_weight=0.75, focal_tversky_weight=0.25,
        alpha=0.40, beta=0.60, gamma=0.75,
    ):
        super().__init__()
        if not math.isclose(
            focal_dice_weight + focal_tversky_weight, 1.0, abs_tol=1e-8
        ):
            raise ValueError("Composite loss weights must sum to 1.")
        self.focal_dice_weight = focal_dice_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.focal_dice = FocalDiceLoss(
            focal_alpha=0.75, focal_gamma=2.0,
            focal_weight=0.5, dice_weight=0.5,
        )
        self.focal_tversky = FocalTverskyLoss(
            alpha=alpha, beta=beta, gamma=gamma,
        )

    def components(self, logits, targets):
        # The model may emit float16 logits under AMP. Both loss components,
        # including BCE and their reductions, must run in float32 to prevent
        # overflow/underflow after several epochs of increasingly large logits.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits_float = logits.float()
            targets_float = targets.float()
            stable = self.focal_dice(logits_float, targets_float)
            recall = self.focal_tversky(logits_float, targets_float)
            total = (
                self.focal_dice_weight * stable
                + self.focal_tversky_weight * recall
            )
        return total, stable, recall

    def forward(self, logits, targets):
        return self.components(logits, targets)[0]


""" + model_suffix

preflight_markdown = new_markdown_cell("""### 5. Preflight numerical and gradient checks

Before creating the model, test empty masks, tiny lesions and larger lesions
across very negative, neutral and very positive logits. Training is blocked if
any loss or gradient is non-finite, or if positive examples have no gradient.
""")

preflight_code = new_code_cell(r'''preflight_loss = StableCompositeLoss(
    focal_dice_weight=FOCAL_DICE_WEIGHT,
    focal_tversky_weight=FOCAL_TVERSKY_WEIGHT,
    alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
    gamma=FOCAL_TVERSKY_GAMMA,
)
preflight_rows = []
for case, lesion_side in [("empty", 0), ("tiny", 2), ("medium", 8), ("large", 16)]:
    target = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    if lesion_side:
        start = (32 - lesion_side) // 2
        target[:, :, start:start + lesion_side, start:start + lesion_side] = 1.0
    for logit_level in [-12.0, 0.0, 12.0]:
        logits = torch.full(
            target.shape, logit_level, dtype=torch.float32, requires_grad=True
        )
        total, stable, recall = preflight_loss.components(logits, target)
        total.backward()
        gradient_norm = float(logits.grad.abs().sum())
        row = {
            "case": case, "logit_level": logit_level,
            "total_loss": float(total.detach()),
            "focal_dice": float(stable.detach()),
            "focal_tversky": float(recall.detach()),
            "gradient_l1": gradient_norm,
        }
        if not all(np.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise FloatingPointError(f"Non-finite preflight result: {row}")
        if lesion_side and gradient_norm <= 0:
            raise FloatingPointError(f"Missing positive-case gradient: {row}")
        preflight_rows.append(row)

preflight_table = pd.DataFrame(preflight_rows)
display(preflight_table.style.format({
    "total_loss": "{:.6f}", "focal_dice": "{:.6f}",
    "focal_tversky": "{:.6f}", "gradient_l1": "{:.6e}",
}))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for case, group in preflight_table.groupby("case"):
    axes[0].plot(group["logit_level"], group["total_loss"], marker="o", label=case)
    axes[1].plot(group["logit_level"], group["gradient_l1"], marker="o", label=case)
axes[0].set_title("Composite loss preflight")
axes[0].set_xlabel("Uniform logit"); axes[0].set_ylabel("Loss"); axes[0].legend()
axes[1].set_title("Composite gradient magnitude")
axes[1].set_xlabel("Uniform logit"); axes[1].set_ylabel("Gradient L1")
axes[1].set_yscale("log"); axes[1].legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "composite_loss_preflight.png", dpi=160, bbox_inches="tight")
plt.show()
print("PASS: composite loss is finite with non-zero positive-case gradients.")
''')
notebook.cells[loss_index + 1:loss_index + 1] = [
    preflight_markdown, preflight_code
]

training_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code" and "criterion = FocalTverskyLoss(" in cell.source
)
training_cell = notebook.cells[training_index].source
training_cell = training_cell.replace(
    """criterion = FocalTverskyLoss(
    alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
    gamma=FOCAL_TVERSKY_GAMMA,
)""",
    """criterion = StableCompositeLoss(
    focal_dice_weight=FOCAL_DICE_WEIGHT,
    focal_tversky_weight=FOCAL_TVERSKY_WEIGHT,
    alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
    gamma=FOCAL_TVERSKY_GAMMA,
)""",
)
notebook.cells[training_index].source = training_cell

comparison_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code"
    and "loss_ablation_comparison_dashboard.png" in cell.source
)
notebook.cells[comparison_index - 1].source = """### Four-experiment comparison dashboard

Compare the stabilized composite candidate against the trusted Focal-Dice
baseline, failed balanced sampler and collapsed pure Focal-Tversky run.
"""
notebook.cells[comparison_index].source = r'''baseline_history = pd.read_csv(
    BASELINE_OUTPUT_DIR / "patient_aware_history.csv"
)
sampler_history = pd.read_csv(
    SAMPLER_OUTPUT_DIR / "patient_aware_history.csv"
)
pure_history = pd.read_csv(
    PURE_TVERSKY_OUTPUT_DIR / "patient_aware_history.csv"
)

if not history_frame.empty and not best_patient_metrics.empty:
    histories = {
        "Focal-Dice baseline": baseline_history,
        "Balanced sampler": sampler_history,
        "Pure Focal-Tversky": pure_history,
        "Stable composite": history_frame,
    }
    best_rows = {
        name: frame.loc[frame["val_mean_patient_dice"].idxmax()]
        for name, frame in histories.items()
    }
    comparison = pd.DataFrame([
        {
            "experiment": name,
            "mean_patient_dice": row["val_mean_patient_dice"],
            "global_dice": row["val_global_micro_dice"],
            "precision": row["val_pixel_precision"],
            "recall": row["val_pixel_recall"],
            "positive_empty_pct": row["val_positive_predicted_empty_pct"],
            "empty_fp_pct": row["val_empty_slice_false_positive_pct"],
        }
        for name, row in best_rows.items()
    ])

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    styles = {
        "Focal-Dice baseline": ("o", "#8A949E"),
        "Balanced sampler": ("^", "#E68632"),
        "Pure Focal-Tversky": ("x", "#C2A33A"),
        "Stable composite": ("s", "#2878B5"),
    }
    for name, frame in histories.items():
        marker, color = styles[name]
        axes[0, 0].plot(
            frame["epoch"], frame["val_mean_patient_dice"],
            marker=marker, color=color, label=name,
        )
    axes[0, 0].set_title("Mean patient Dice by epoch")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylim(0, 1); axes[0, 0].legend()

    x = np.arange(len(comparison))
    axes[0, 1].bar(
        x, comparison["mean_patient_dice"],
        color=[styles[name][1] for name in comparison["experiment"]],
        edgecolor="#333333",
    )
    axes[0, 1].axhline(
        best_rows["Focal-Dice baseline"]["val_mean_patient_dice"],
        linestyle="--", color="#333333", label="Baseline best",
    )
    axes[0, 1].set_xticks(x, comparison["experiment"], rotation=12)
    axes[0, 1].set_title("Best mean patient Dice")
    axes[0, 1].set_ylim(0, 1); axes[0, 1].legend()

    axes[1, 0].bar(
        x - 0.18, comparison["precision"], 0.36,
        label="Precision", color="#B9C2CC", edgecolor="#333333",
    )
    axes[1, 0].bar(
        x + 0.18, comparison["recall"], 0.36,
        label="Recall", color="#2878B5", edgecolor="#333333",
    )
    axes[1, 0].set_xticks(x, comparison["experiment"], rotation=12)
    axes[1, 0].set_title("Pixel precision and recall")
    axes[1, 0].set_ylim(0, 1); axes[1, 0].legend()

    for row in comparison.itertuples(index=False):
        axes[1, 1].scatter(
            row.positive_empty_pct, row.empty_fp_pct, s=120,
            color=styles[row.experiment][1], marker=styles[row.experiment][0],
            label=row.experiment,
        )
    axes[1, 1].axvline(30, linestyle="--", color="#333333")
    axes[1, 1].axhline(15, linestyle="--", color="#333333")
    axes[1, 1].set_title("Detection-error guardrails")
    axes[1, 1].set_xlabel("Positive slices predicted empty (%)")
    axes[1, 1].set_ylabel("Empty slices with false positives (%)")
    axes[1, 1].legend()

    fig.suptitle("Stabilized composite-loss comparison", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "composite_loss_comparison_dashboard.png", dpi=160, bbox_inches="tight")
    plt.show()
    display(comparison.style.format({
        column: "{:.4f}" for column in comparison.columns if column != "experiment"
    }))
else:
    print("Comparison is available after training and best-checkpoint evaluation.")
'''

takeaways_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "markdown" and cell.source.startswith("## Takeaways")
)
notebook.cells[takeaways_index].source = """## Takeaways

Promote the composite objective only if it beats the Focal-Dice baseline on mean
patient Dice, recovers patient 116, improves smallest-lesion detection and stays
inside both detection-error guardrails.
"""

gate_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code"
    and "recall_aware_loss_gate_result.json" in cell.source
)
notebook.cells[gate_index].source = r'''minimum_decision_epoch_reached = (
    not history_frame.empty
    and int(history_frame["epoch"].max()) >= MIN_EPOCHS_BEFORE_EARLY_STOP
)
baseline_gate = json.loads(
    (BASELINE_OUTPUT_DIR / "patient_aware_gate_result.json").read_text(encoding="utf-8")
)
pure_gate = json.loads(
    (PURE_TVERSKY_OUTPUT_DIR / "recall_aware_loss_gate_result.json").read_text(encoding="utf-8")
)

if not minimum_decision_epoch_reached:
    completed_epochs = int(history_frame["epoch"].max()) if not history_frame.empty else 0
    final_gate = {
        "status": "incomplete_continue_to_epoch_10",
        "epochs_completed": completed_epochs,
        "minimum_decision_epoch": MIN_EPOCHS_BEFORE_EARLY_STOP,
        "loss_strategy": LOSS_STRATEGY,
        "test_images_accessed": False,
        "decision": f"INCOMPLETE — continue from epoch {completed_epochs + 1}.",
    }
elif best_patient_metrics.empty or size_metrics.empty:
    final_gate = {
        "status": "not_run", "loss_strategy": LOSS_STRATEGY,
        "test_images_accessed": False,
        "decision": "Run training and best-checkpoint evaluation.",
    }
else:
    best_row = history_frame.loc[history_frame["val_mean_patient_dice"].idxmax()]
    positive_patients = best_patient_metrics.loc[best_patient_metrics["true_pixels"].gt(0)]
    volume_104 = positive_patients.loc[positive_patients["volume_id"].eq(104), "micro_dice"]
    volume_116 = positive_patients.loc[positive_patients["volume_id"].eq(116), "micro_dice"]
    q1 = size_metrics.loc[size_metrics["size_quartile"].eq("Q1 smallest")].iloc[0]

    mean_improved = float(best_row["val_mean_patient_dice"]) > float(
        baseline_gate["best_mean_patient_dice"]
    )
    volume_104_preserved = bool(
        len(volume_104)
        and float(volume_104.iloc[0]) >= float(baseline_gate["volume_104_dice"])
    )
    volume_116_recovered = bool(len(volume_116) and float(volume_116.iloc[0]) > 0.05)
    q1_detection_improved = float(q1["detected_pct"]) > 35.0
    guardrails = bool(
        float(best_row["val_positive_predicted_empty_pct"]) < 30.0
        and float(best_row["val_empty_slice_false_positive_pct"]) < 15.0
    )
    noncollapsed = bool(
        float(best_row["val_positive_slice_recall"]) > 0.0
        and float(best_row["val_global_micro_dice"]) > 0.0
    )
    ready = all([
        mean_improved, volume_104_preserved, volume_116_recovered,
        q1_detection_improved, guardrails, noncollapsed,
    ])
    final_gate = {
        "status": "composite_loss_pass" if ready else "composite_loss_fail",
        "manifest_sha256": manifest_hash,
        "loss_strategy": LOSS_STRATEGY,
        "epochs_completed": int(history_frame["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "best_global_micro_dice": float(best_row["val_global_micro_dice"]),
        "best_mean_patient_dice": float(best_row["val_mean_patient_dice"]),
        "baseline_mean_patient_dice": float(baseline_gate["best_mean_patient_dice"]),
        "pure_tversky_mean_patient_dice": float(pure_gate["best_mean_patient_dice"]),
        "volume_104_dice": float(volume_104.iloc[0]) if len(volume_104) else None,
        "volume_116_dice": float(volume_116.iloc[0]) if len(volume_116) else None,
        "q1_smallest_detected_pct": float(q1["detected_pct"]),
        "positive_predicted_empty_pct": float(best_row["val_positive_predicted_empty_pct"]),
        "empty_slice_false_positive_pct": float(best_row["val_empty_slice_false_positive_pct"]),
        "noncollapsed": noncollapsed,
        "mean_patient_improved": mean_improved,
        "volume_104_preserved": volume_104_preserved,
        "volume_116_recovered": volume_116_recovered,
        "q1_detection_improved": q1_detection_improved,
        "detection_guardrails_pass": guardrails,
        "test_images_accessed": False,
        "decision": (
            "PASS — confirm the composite objective with a longer controlled run."
            if ready else
            "FAIL — retain Focal-Dice and move to appearance/domain robustness diagnostics."
        ),
    }

(OUTPUT_DIR / "composite_loss_gate_result.json").write_text(
    json.dumps(final_gate, indent=2), encoding="utf-8"
)
display(pd.DataFrame([final_gate]).T.rename(columns={0: "result"}))
print(final_gate["decision"])
'''

for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

notebook.metadata["experiment"] = {
    "name": "stabilized_composite_loss_ablation",
    "single_intervention": "loss_function",
    "test_split_locked": True,
    "candidate": "0.75 FocalDice plus 0.25 safe FocalTversky",
}

nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")

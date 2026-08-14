"""Build the controlled recall-aware focal-Tversky loss ablation notebook."""

from copy import deepcopy
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "patient_lesion_balanced_sampler_ablation.ipynb"
OUTPUT = ROOT / "Practice" / "recall_aware_focal_tversky_loss_ablation.ipynb"

notebook = deepcopy(nbformat.read(SOURCE, as_version=4))

notebook.cells[0].source = """# Recall-Aware Focal-Tversky Loss Ablation

This controlled experiment returns to the stronger baseline sampler and changes
only the loss function. The goal is to recover missed tumor pixels and patient
116 without recreating the balanced sampler's false-positive burden.

Dataset, split, preprocessing, architecture, initialization, augmentation,
sampler, optimizer, learning-rate schedule and validation protocol remain fixed.
The test split remains locked.
"""

notebook.cells[1].source = """## tl;dr

The balanced sampler doubled smallest-lesion detection from **25.5% to 54.0%**,
but mean patient Dice fell from **0.3831 to 0.3562** and empty-slice false
positives rose from **3.35% to 17.83%**.

This fresh 10-epoch experiment restores the baseline `3x` positive-slice sampler
and replaces Focal-Dice with a moderately recall-aware Focal-Tversky loss:

- false-positive penalty `alpha = 0.40`
- false-negative penalty `beta = 0.60`
- focal exponent `gamma = 0.75`

Run all cells once with the project `.venv` kernel.
"""

notebook.cells[2].source = """## Context & Methods

### Key assumptions

- The manifest, verified loader, tumor masks, orientation and splits are frozen.
- Sampling uses the original baseline `3x` positive-slice weight.
- Validation patients never influence training or checkpoint selection beyond
  the already-defined mean-patient-Dice criterion.
- A fresh model provides a fair loss comparison.

### Single intervention

Focal-Tversky penalizes false negatives more strongly than false positives,
using a moderate `0.60 / 0.40` asymmetry. This is intentionally less aggressive
than the failed balanced sampler. Promotion still requires false positives below
15%.
"""

setup = notebook.cells[3].source
setup = setup.replace(
    'BASELINE_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"\n'
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_lesion_balanced_outputs"',
    'BASELINE_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"\n'
    'SAMPLER_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_lesion_balanced_outputs"\n'
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "recall_aware_loss_outputs"',
)
setup = setup.replace(
    'SAMPLER_STRATEGY = "inverse_volume_x_lesion_quartile_v1"\n'
    "LESION_FACTORS = {0: 1.0, 1: 8.0, 2: 5.0, 3: 3.0, 4: 2.0}\n"
    "MIN_NORMALIZED_WEIGHT = 0.20\n"
    "MAX_NORMALIZED_WEIGHT = 8.00",
    'SAMPLER_STRATEGY = "positive_slice_weight_3_baseline"\n'
    'LOSS_STRATEGY = "focal_tversky_a040_b060_g075"\n'
    "POSITIVE_SAMPLE_WEIGHT = 3.0\n"
    "TVERSKY_ALPHA = 0.40\n"
    "TVERSKY_BETA = 0.60\n"
    "FOCAL_TVERSKY_GAMMA = 0.75",
)
notebook.cells[3].source = setup

notebook.cells[6].source = """## Data

The same verified manifest and natural validation loader are used. The training
sampler is restored exactly to the baseline protocol: tumor-positive slices
receive weight `3`, and empty slices receive weight `1`.
"""

notebook.cells[10].source = "### 3. Restore the baseline weighted training sampler"
notebook.cells[11].source = r'''train_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="train", root_dir=DATASET_ROOT,
    target="tumor", transform=train_transform, validate_paths=True,
)
val_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="val", root_dir=DATASET_ROOT,
    target="tumor", transform=None, validate_paths=True,
)

train_flags = np.asarray(train_dataset.tumor_positive_flags, dtype=bool)
weights = np.where(train_flags, POSITIVE_SAMPLE_WEIGHT, 1.0)
sampler_generator = torch.Generator().manual_seed(SEED)
train_sampler = WeightedRandomSampler(
    torch.as_tensor(weights, dtype=torch.double),
    num_samples=len(train_dataset), replacement=True,
    generator=sampler_generator,
)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
)

natural_rate = float(train_flags.mean())
expected_rate = (
    POSITIVE_SAMPLE_WEIGHT * natural_rate
    / (POSITIVE_SAMPLE_WEIGHT * natural_rate + 1 - natural_rate)
)
preview_indices = np.asarray(list(iter(train_sampler))[: min(12000, len(train_dataset))])
preview_rate = float(train_flags[preview_indices].mean())
sampling_audit = pd.DataFrame({
    "measure": ["Natural training", "Expected weighted", "Sampler preview"],
    "positive_slice_pct": [100 * natural_rate, 100 * expected_rate, 100 * preview_rate],
})
display(sampling_audit.style.format({"positive_slice_pct": "{:.2f}%"}))
print(f"Train={len(train_dataset):,} | Validation={len(val_dataset):,}")
'''

notebook.cells[12].source = "### 4. Verify the restored sampler and paired augmentation"
notebook.cells[13].source = r'''fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].bar(
    sampling_audit["measure"], sampling_audit["positive_slice_pct"],
    color=["#B9C2CC", "#E6A23C", "#2878B5"], edgecolor="#333333",
)
axes[0].set_title("Positive-slice sampling rate")
axes[0].set_ylabel("Share (%)")
axes[0].tick_params(axis="x", rotation=20)

positive_indices = np.flatnonzero(train_flags)
preview_source_indices = np.random.default_rng(SEED).choice(
    positive_indices, size=min(3, len(positive_indices)), replace=False
)
for axis, source_index in zip(axes[1:], preview_source_indices):
    sample = train_dataset[int(source_index)]
    image = sample["image"][0].numpy()
    mask = sample["mask"][0].numpy()
    axis.imshow(image, cmap="gray")
    axis.contour(mask, levels=[0.5], colors=["#E15759"], linewidths=1.0)
    axis.set_title(f"{sample['sample_id']} | pixels={int(mask.sum()):,}")
    axis.axis("off")

fig.suptitle("Restored baseline sampler and augmentation QA", fontsize=16)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sampler_and_augmentation_audit.png", dpi=160, bbox_inches="tight")
plt.show()
'''

loss_cell = notebook.cells[15].source
loss_cell = loss_cell.replace(
    "from src.framework.losses.focal_dice import FocalDiceLoss\n",
    "import torch.nn as nn\n\n"
    "class FocalTverskyLoss(nn.Module):\n"
    "    \"\"\"Batch-mean focal Tversky loss with explicit FP/FN penalties.\"\"\"\n"
    "    def __init__(self, alpha=0.40, beta=0.60, gamma=0.75, smooth=1.0):\n"
    "        super().__init__()\n"
    "        if not math.isclose(alpha + beta, 1.0, abs_tol=1e-8):\n"
    "            raise ValueError(\"Tversky alpha and beta must sum to 1.\")\n"
    "        self.alpha, self.beta = alpha, beta\n"
    "        self.gamma, self.smooth = gamma, smooth\n\n"
    "    def forward(self, logits, targets):\n"
    "        # Accumulate overlap terms in float32 even under AMP. Fractional\n"
    "        # powers of a tiny negative value caused by fp16 rounding produce NaN.\n"
    "        probabilities = torch.sigmoid(logits.float())\n"
    "        targets = targets.float()\n"
    "        dimensions = tuple(range(1, logits.ndim))\n"
    "        true_positive = (probabilities * targets).sum(dim=dimensions)\n"
    "        false_positive = (probabilities * (1.0 - targets)).sum(dim=dimensions)\n"
    "        false_negative = ((1.0 - probabilities) * targets).sum(dim=dimensions)\n"
    "        score = (true_positive + self.smooth) / (\n"
    "            true_positive + self.alpha * false_positive\n"
    "            + self.beta * false_negative + self.smooth\n"
    "        )\n"
    "        score = score.clamp(min=0.0, max=1.0)\n"
    "        focal_error = (1.0 - score).clamp(min=0.0, max=1.0)\n"
    "        return torch.pow(focal_error, self.gamma).mean()\n\n",
)
notebook.cells[15].source = loss_cell

training_cell = notebook.cells[17].source
old_criterion = """criterion = FocalDiceLoss(
    focal_alpha=0.75, focal_gamma=2.0, focal_weight=0.5, dice_weight=0.5
)"""
new_criterion = """criterion = FocalTverskyLoss(
    alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA,
    gamma=FOCAL_TVERSKY_GAMMA,
)"""
if old_criterion not in training_cell:
    raise RuntimeError("Could not locate the baseline criterion construction.")
training_cell = training_cell.replace(old_criterion, new_criterion)
notebook.cells[17].source = training_cell

for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace(
        '"sampler_strategy": SAMPLER_STRATEGY,',
        '"sampler_strategy": SAMPLER_STRATEGY,\n'
        '        "loss_strategy": LOSS_STRATEGY,',
    )

# Replace the inherited sampler-specific comparison with a three-experiment view.
comparison_markdown_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "markdown"
    and cell.source.startswith("### Baseline comparison dashboard")
)
notebook.cells[comparison_markdown_index].source = """### Controlled loss comparison dashboard

Compare the recall-aware loss with the frozen Focal-Dice baseline and the
rejected balanced-sampler experiment using validation metrics only.
"""
notebook.cells[comparison_markdown_index + 1] = new_code_cell(r'''baseline_history = pd.read_csv(
    BASELINE_OUTPUT_DIR / "patient_aware_history.csv"
)
sampler_history = pd.read_csv(
    SAMPLER_OUTPUT_DIR / "patient_aware_history.csv"
)
baseline_patients = pd.read_csv(
    BASELINE_OUTPUT_DIR / "best_validation_patient_metrics.csv"
)
sampler_patients = pd.read_csv(
    SAMPLER_OUTPUT_DIR / "best_validation_patient_metrics.csv"
)

if not history_frame.empty and not best_patient_metrics.empty:
    baseline_best = baseline_history.loc[baseline_history["val_mean_patient_dice"].idxmax()]
    sampler_best = sampler_history.loc[sampler_history["val_mean_patient_dice"].idxmax()]
    loss_best = history_frame.loc[history_frame["val_mean_patient_dice"].idxmax()]
    experiments = {
        "Focal-Dice baseline": baseline_best,
        "Balanced sampler": sampler_best,
        "Focal-Tversky": loss_best,
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
        for name, row in experiments.items()
    ])

    patient_comparison = (
        baseline_patients[["volume_id", "micro_dice"]]
        .rename(columns={"micro_dice": "Focal-Dice baseline"})
        .merge(
            sampler_patients[["volume_id", "micro_dice"]].rename(
                columns={"micro_dice": "Balanced sampler"}
            ), on="volume_id",
        )
        .merge(
            best_patient_metrics[["volume_id", "micro_dice"]].rename(
                columns={"micro_dice": "Focal-Tversky"}
            ), on="volume_id",
        )
    )
    patient_comparison = patient_comparison.loc[
        patient_comparison["volume_id"].isin([104, 107, 108, 109, 110, 111, 112, 113, 116])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    for frame, label, marker, color in [
        (baseline_history, "Focal-Dice baseline", "o", "#8A949E"),
        (sampler_history, "Balanced sampler", "^", "#E68632"),
        (history_frame, "Focal-Tversky", "s", "#2878B5"),
    ]:
        axes[0, 0].plot(
            frame["epoch"], frame["val_mean_patient_dice"],
            marker=marker, color=color, label=label,
        )
    axes[0, 0].set_title("Mean patient Dice by epoch")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylim(0, 1); axes[0, 0].legend()

    x = np.arange(len(patient_comparison))
    width = 0.26
    for offset, column, color in [
        (-width, "Focal-Dice baseline", "#B9C2CC"),
        (0, "Balanced sampler", "#E68632"),
        (width, "Focal-Tversky", "#2878B5"),
    ]:
        axes[0, 1].bar(
            x + offset, patient_comparison[column], width,
            label=column, color=color, edgecolor="#333333",
        )
    axes[0, 1].set_xticks(x, patient_comparison["volume_id"].astype(str))
    axes[0, 1].set_title("Tumor-positive patient Dice")
    axes[0, 1].set_xlabel("Validation volume"); axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    metric_x = np.arange(len(comparison))
    axes[1, 0].bar(
        metric_x - 0.18, comparison["precision"], 0.36,
        label="Precision", color="#B9C2CC", edgecolor="#333333",
    )
    axes[1, 0].bar(
        metric_x + 0.18, comparison["recall"], 0.36,
        label="Recall", color="#2878B5", edgecolor="#333333",
    )
    axes[1, 0].set_xticks(metric_x, comparison["experiment"], rotation=12)
    axes[1, 0].set_title("Pixel precision and recall")
    axes[1, 0].set_ylim(0, 1); axes[1, 0].legend()

    axes[1, 1].scatter(
        comparison["positive_empty_pct"], comparison["empty_fp_pct"],
        s=120, color=["#8A949E", "#E68632", "#2878B5"], edgecolor="#333333",
    )
    for row in comparison.itertuples(index=False):
        axes[1, 1].annotate(
            row.experiment, (row.positive_empty_pct, row.empty_fp_pct),
            xytext=(5, 5), textcoords="offset points",
        )
    axes[1, 1].axvline(30, linestyle="--", color="#333333")
    axes[1, 1].axhline(15, linestyle="--", color="#333333")
    axes[1, 1].set_title("Detection-error guardrails")
    axes[1, 1].set_xlabel("Positive slices predicted empty (%)")
    axes[1, 1].set_ylabel("Empty slices with false positives (%)")

    fig.suptitle("Controlled recall-aware loss comparison", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_ablation_comparison_dashboard.png", dpi=160, bbox_inches="tight")
    plt.show()
    display(comparison.style.format({
        column: "{:.4f}" for column in comparison.columns if column != "experiment"
    }))
else:
    print("Comparison is available after training and best-checkpoint evaluation.")
''')

takeaways_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "markdown" and cell.source.startswith("## Takeaways")
)
notebook.cells[takeaways_index].source = """## Takeaways

This is a controlled loss ablation. Promote Focal-Tversky only if it improves
mean patient Dice and patient 116 while keeping empty-slice false positives
below 15%. Global Dice alone is insufficient.
"""

gate_index = next(
    index for index, cell in enumerate(notebook.cells)
    if cell.cell_type == "code" and "sampler_ablation_gate_result.json" in cell.source
)
notebook.cells[gate_index].source = r'''minimum_decision_epoch_reached = (
    not history_frame.empty
    and int(history_frame["epoch"].max()) >= MIN_EPOCHS_BEFORE_EARLY_STOP
)
baseline_gate = json.loads(
    (BASELINE_OUTPUT_DIR / "patient_aware_gate_result.json").read_text(encoding="utf-8")
)
sampler_gate = json.loads(
    (SAMPLER_OUTPUT_DIR / "sampler_ablation_gate_result.json").read_text(encoding="utf-8")
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
        "status": "not_run",
        "loss_strategy": LOSS_STRATEGY,
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
    ready = all([
        mean_improved, volume_104_preserved, volume_116_recovered,
        q1_detection_improved, guardrails,
    ])
    final_gate = {
        "status": "recall_aware_loss_pass" if ready else "recall_aware_loss_fail",
        "manifest_sha256": manifest_hash,
        "sampler_strategy": SAMPLER_STRATEGY,
        "loss_strategy": LOSS_STRATEGY,
        "epochs_completed": int(history_frame["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "best_global_micro_dice": float(best_row["val_global_micro_dice"]),
        "best_mean_patient_dice": float(best_row["val_mean_patient_dice"]),
        "baseline_mean_patient_dice": float(baseline_gate["best_mean_patient_dice"]),
        "sampler_mean_patient_dice": float(sampler_gate["best_mean_patient_dice"]),
        "volume_104_dice": float(volume_104.iloc[0]) if len(volume_104) else None,
        "volume_116_dice": float(volume_116.iloc[0]) if len(volume_116) else None,
        "q1_smallest_detected_pct": float(q1["detected_pct"]),
        "positive_predicted_empty_pct": float(best_row["val_positive_predicted_empty_pct"]),
        "empty_slice_false_positive_pct": float(best_row["val_empty_slice_false_positive_pct"]),
        "mean_patient_improved": mean_improved,
        "volume_104_preserved": volume_104_preserved,
        "volume_116_recovered": volume_116_recovered,
        "q1_detection_improved": q1_detection_improved,
        "detection_guardrails_pass": guardrails,
        "test_images_accessed": False,
        "decision": (
            "PASS — retain Focal-Tversky and plan a longer confirmation run."
            if ready else
            "FAIL — retain the Focal-Dice baseline and investigate domain/appearance robustness."
        ),
    }

(OUTPUT_DIR / "recall_aware_loss_gate_result.json").write_text(
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
    "name": "recall_aware_focal_tversky_loss_ablation",
    "single_intervention": "loss_function",
    "test_split_locked": True,
    "baseline": "FocalDiceLoss with positive-slice weight 3",
    "candidate": "FocalTverskyLoss alpha 0.40 beta 0.60 gamma 0.75",
}

nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")

"""Build the controlled patient-and-lesion-balanced sampler ablation notebook."""

from copy import deepcopy
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "auto_cool_continue_patient_aware_to_epoch10.ipynb"
OUTPUT = ROOT / "Practice" / "patient_lesion_balanced_sampler_ablation.ipynb"

notebook = deepcopy(nbformat.read(SOURCE, as_version=4))

notebook.cells[0].source = """# Patient-and-Lesion-Balanced Sampler Ablation

This controlled experiment addresses the epoch-10 failure pattern without
changing the dataset, split, preprocessing, architecture, augmentation, loss,
optimizer, learning-rate schedule, or validation protocol.

**Only the training sampler changes.** It balances training volumes and
upweights small positive lesions. The test split remains locked.
"""

notebook.cells[1].source = """## tl;dr

The previous best checkpoint reached mean patient Dice **0.3831** at epoch 9,
but patient 116 remained effectively undetected and 64.6% of the smallest
positive slices were predicted empty.

Run this notebook once with the project `.venv` kernel. It trains a fresh,
controlled 10-epoch sampler ablation and compares it with the frozen baseline.
Automatic GPU cooling and exact checkpoint resume remain enabled.
"""

notebook.cells[2].source = """## Context & Methods

### Key assumptions

- Manifest hash, verified loader, tumor-mask semantics, orientation and splits
  remain frozen.
- Validation patients never influence sampling weights.
- A fresh model is required for a fair sampler comparison.
- The epoch-10 baseline is the comparator; the test split stays inaccessible.

### Intervention

Each training slice receives:

1. an inverse-volume-frequency factor, and
2. a lesion-size factor: empty `1`, Q1 `8`, Q2 `5`, Q3 `3`, Q4 `2`.

Weights are normalized and clipped to prevent a few slices dominating an epoch.
"""

setup = notebook.cells[3].source
setup = setup.replace(
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"',
    'BASELINE_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"\n'
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_lesion_balanced_outputs"',
)
setup = setup.replace(
    "POSITIVE_SAMPLE_WEIGHT = 3.0",
    'SAMPLER_STRATEGY = "inverse_volume_x_lesion_quartile_v1"\n'
    "LESION_FACTORS = {0: 1.0, 1: 8.0, 2: 5.0, 3: 3.0, 4: 2.0}\n"
    "MIN_NORMALIZED_WEIGHT = 0.20\n"
    "MAX_NORMALIZED_WEIGHT = 8.00",
)
notebook.cells[3].source = setup

notebook.cells[6].source = """## Data

The verified manifest is loaded exactly as before. Sampling weights are derived
only from training rows. The natural validation loader remains unweighted and
ordered, so all baseline comparisons use identical validation data.
"""

notebook.cells[10].source = (
    "### 3. Build the patient-and-lesion-balanced training sampler"
)

notebook.cells[11].source = r'''train_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="train", root_dir=DATASET_ROOT,
    target="tumor", transform=train_transform, validate_paths=True,
)
val_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="val", root_dir=DATASET_ROOT,
    target="tumor", transform=None, validate_paths=True,
)

train_rows = pd.DataFrame([
    {
        "sample_id": row["sample_id"],
        "volume_id": int(row["volume_id"]),
        "tumor_pixels": int(row["tumor_pixels"]),
    }
    for row in train_dataset.rows
])
if train_rows["sample_id"].tolist() != train_dataset.sample_ids:
    raise RuntimeError("Training manifest order does not match dataset order.")

positive_mask = train_rows["tumor_pixels"].gt(0)
positive_pixels = train_rows.loc[positive_mask, "tumor_pixels"]
quartile_edges = positive_pixels.quantile([0.25, 0.50, 0.75]).to_numpy()
train_rows["lesion_stratum"] = 0
train_rows.loc[positive_mask, "lesion_stratum"] = (
    np.searchsorted(quartile_edges, positive_pixels.to_numpy(), side="right") + 1
)

volume_counts = train_rows.groupby("volume_id")["sample_id"].transform("count")
inverse_volume_factor = len(train_rows) / (
    train_rows["volume_id"].nunique() * volume_counts
)
lesion_factor = train_rows["lesion_stratum"].map(LESION_FACTORS).astype(float)
raw_weights = inverse_volume_factor * lesion_factor
normalized_weights = raw_weights / raw_weights.mean()
train_rows["sampling_weight"] = normalized_weights.clip(
    MIN_NORMALIZED_WEIGHT, MAX_NORMALIZED_WEIGHT
)

train_flags = positive_mask.to_numpy()
weights = train_rows["sampling_weight"].to_numpy(dtype=np.float64)
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

preview_indices = np.asarray(list(iter(train_sampler))[: min(12000, len(train_dataset))])
natural_distribution = (
    train_rows.groupby("lesion_stratum").size().rename("natural_count")
)
sampled_distribution = (
    train_rows.iloc[preview_indices].groupby("lesion_stratum").size().rename("sampled_count")
)
sampling_audit = pd.concat(
    [natural_distribution, sampled_distribution], axis=1
).fillna(0)
sampling_audit["natural_pct"] = 100 * sampling_audit["natural_count"] / sampling_audit["natural_count"].sum()
sampling_audit["sampled_pct"] = 100 * sampling_audit["sampled_count"] / sampling_audit["sampled_count"].sum()
sampling_audit.index = ["empty", "Q1 smallest", "Q2", "Q3", "Q4 largest"]

volume_audit = train_rows.groupby("volume_id").agg(
    slices=("sample_id", "size"),
    positives=("tumor_pixels", lambda values: int((values > 0).sum())),
    mean_weight=("sampling_weight", "mean"),
)
sampled_volume_share = (
    train_rows.iloc[preview_indices]["volume_id"].value_counts(normalize=True)
    .mul(100).rename("sampled_pct")
)
volume_audit = volume_audit.join(sampled_volume_share).fillna({"sampled_pct": 0})

display(sampling_audit.style.format({
    "natural_pct": "{:.2f}%", "sampled_pct": "{:.2f}%"
}))
print(
    f"Train={len(train_dataset):,} | Validation={len(val_dataset):,} | "
    f"training volumes={train_rows['volume_id'].nunique()}"
)
'''

notebook.cells[12].source = "### 4. Visualize sampler behavior and augmentation integrity"
notebook.cells[13].source = r'''fig, axes = plt.subplots(2, 3, figsize=(18, 11))

sampling_audit[["natural_pct", "sampled_pct"]].plot.bar(
    ax=axes[0, 0], color=["#B9C2CC", "#2878B5"], edgecolor="#333333"
)
axes[0, 0].set_title("Natural versus sampled lesion strata")
axes[0, 0].set_xlabel("Training slice stratum"); axes[0, 0].set_ylabel("Share (%)")
axes[0, 0].legend(["Natural", "Sampler preview"])

axes[0, 1].scatter(
    volume_audit["slices"], volume_audit["sampled_pct"],
    s=35 + 2 * np.sqrt(volume_audit["positives"]), alpha=0.75,
    color="#E68632", edgecolor="#333333",
)
axes[0, 1].set_title("Volume balancing audit")
axes[0, 1].set_xlabel("Natural slices per volume")
axes[0, 1].set_ylabel("Sampled share (%)")

weight_by_stratum = train_rows.groupby("lesion_stratum")["sampling_weight"].agg(
    ["median", "min", "max"]
)
axes[0, 2].errorbar(
    np.arange(len(weight_by_stratum)), weight_by_stratum["median"],
    yerr=[
        weight_by_stratum["median"] - weight_by_stratum["min"],
        weight_by_stratum["max"] - weight_by_stratum["median"],
    ],
    fmt="o", capsize=4, color="#2878B5", ecolor="#8A949E",
)
axes[0, 2].set_xticks(
    np.arange(len(weight_by_stratum)),
    ["empty", "Q1", "Q2", "Q3", "Q4"],
)
axes[0, 2].set_title("Normalized sampling-weight range")
axes[0, 2].set_xlabel("Lesion stratum"); axes[0, 2].set_ylabel("Weight")

positive_indices = np.flatnonzero(train_flags)
preview_source_indices = np.random.default_rng(SEED).choice(
    positive_indices, size=min(3, len(positive_indices)), replace=False
)
for axis, source_index in zip(axes[1], preview_source_indices):
    sample = train_dataset[int(source_index)]
    image = sample["image"][0].numpy()
    mask = sample["mask"][0].numpy()
    axis.imshow(image, cmap="gray")
    axis.contour(mask, levels=[0.5], colors=["#E15759"], linewidths=1.0)
    axis.set_title(
        f"{sample['sample_id']} | tumor pixels={int(mask.sum()):,}"
    )
    axis.axis("off")

fig.suptitle("Patient-and-lesion-balanced sampler QA", fontsize=16)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sampler_audit.png", dpi=160, bbox_inches="tight")
plt.show()
'''

for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace(
        '"positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,',
        '"sampler_strategy": SAMPLER_STRATEGY,',
    )
    cell.source = cell.source.replace(
        "threshold_table = pd.DataFrame()\n\nif best_checkpoint.is_file():",
        "threshold_table = pd.DataFrame()\n"
        "size_metrics = pd.DataFrame()\n\n"
        "if best_checkpoint.is_file():",
    )

notebook.cells[30].source = """## Takeaways

This is a controlled sampler ablation. Do not promote the model because global
Dice alone improves. The sampler must improve the mean patient score, patient
116, and smallest-lesion detection without breaking the false-positive
guardrail.
"""

notebook.cells[31].source = r'''minimum_decision_epoch_reached = (
    not history_frame.empty
    and int(history_frame["epoch"].max()) >= MIN_EPOCHS_BEFORE_EARLY_STOP
)
baseline_gate_path = BASELINE_OUTPUT_DIR / "patient_aware_gate_result.json"
baseline_gate = json.loads(baseline_gate_path.read_text(encoding="utf-8"))

if not minimum_decision_epoch_reached:
    completed_epochs = int(history_frame["epoch"].max()) if not history_frame.empty else 0
    final_gate = {
        "status": "incomplete_continue_to_epoch_10",
        "epochs_completed": completed_epochs,
        "minimum_decision_epoch": MIN_EPOCHS_BEFORE_EARLY_STOP,
        "sampler_strategy": SAMPLER_STRATEGY,
        "test_images_accessed": False,
        "decision": f"INCOMPLETE — continue from epoch {completed_epochs + 1}.",
    }
elif best_patient_metrics.empty or size_metrics.empty:
    final_gate = {
        "status": "not_run",
        "sampler_strategy": SAMPLER_STRATEGY,
        "test_images_accessed": False,
        "decision": "Run training and best-checkpoint evaluation.",
    }
else:
    best_row = history_frame.loc[history_frame["val_mean_patient_dice"].idxmax()]
    positive_patients = best_patient_metrics.loc[best_patient_metrics["true_pixels"].gt(0)]
    volume_104 = positive_patients.loc[positive_patients["volume_id"].eq(104), "micro_dice"]
    volume_116 = positive_patients.loc[positive_patients["volume_id"].eq(116), "micro_dice"]
    q1 = size_metrics.loc[
        size_metrics["size_quartile"].eq("Q1 smallest")
    ].iloc[0]

    mean_improved = float(best_row["val_mean_patient_dice"]) > float(
        baseline_gate["best_mean_patient_dice"]
    )
    volume_104_preserved = bool(
        len(volume_104)
        and float(volume_104.iloc[0]) >= float(baseline_gate["volume_104_dice"])
    )
    volume_116_recovered = bool(len(volume_116) and float(volume_116.iloc[0]) > 0.05)
    q1_detection_improved = float(q1["detected_pct"]) > 35.0
    detection_guardrails = bool(
        float(best_row["val_positive_predicted_empty_pct"]) < 30.0
        and float(best_row["val_empty_slice_false_positive_pct"]) < 15.0
    )
    ready = all([
        mean_improved, volume_104_preserved, volume_116_recovered,
        q1_detection_improved, detection_guardrails,
    ])
    final_gate = {
        "status": "sampler_ablation_pass" if ready else "sampler_ablation_fail",
        "manifest_sha256": manifest_hash,
        "sampler_strategy": SAMPLER_STRATEGY,
        "epochs_completed": int(history_frame["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "best_global_micro_dice": float(best_row["val_global_micro_dice"]),
        "best_mean_patient_dice": float(best_row["val_mean_patient_dice"]),
        "baseline_mean_patient_dice": float(baseline_gate["best_mean_patient_dice"]),
        "volume_104_dice": float(volume_104.iloc[0]) if len(volume_104) else None,
        "volume_116_dice": float(volume_116.iloc[0]) if len(volume_116) else None,
        "q1_smallest_detected_pct": float(q1["detected_pct"]),
        "positive_predicted_empty_pct": float(best_row["val_positive_predicted_empty_pct"]),
        "empty_slice_false_positive_pct": float(best_row["val_empty_slice_false_positive_pct"]),
        "mean_patient_improved": mean_improved,
        "volume_104_preserved": volume_104_preserved,
        "volume_116_recovered": volume_116_recovered,
        "q1_detection_improved": q1_detection_improved,
        "detection_guardrails_pass": detection_guardrails,
        "test_images_accessed": False,
        "decision": (
            "PASS — retain balanced sampling for the next loss ablation."
            if ready else
            "FAIL — balanced sampling alone is insufficient; keep the better baseline and test a recall-aware loss."
        ),
    }

(OUTPUT_DIR / "sampler_ablation_gate_result.json").write_text(
    json.dumps(final_gate, indent=2), encoding="utf-8"
)
display(pd.DataFrame([final_gate]).T.rename(columns={0: "result"}))
print(final_gate["decision"])
'''

comparison_cells = [
    new_markdown_cell("""### Baseline comparison dashboard

Compare the sampler ablation with the frozen epoch-10 baseline using validation
metrics only. The test split remains locked.
"""),
    new_code_cell(r'''baseline_history = pd.read_csv(
    BASELINE_OUTPUT_DIR / "patient_aware_history.csv"
)
baseline_patients = pd.read_csv(
    BASELINE_OUTPUT_DIR / "best_validation_patient_metrics.csv"
)

if not history_frame.empty and not best_patient_metrics.empty:
    baseline_best = baseline_history.loc[
        baseline_history["val_mean_patient_dice"].idxmax()
    ]
    ablation_best = history_frame.loc[
        history_frame["val_mean_patient_dice"].idxmax()
    ]
    metric_comparison = pd.DataFrame({
        "metric": [
            "Global micro Dice", "Mean patient Dice", "Median patient Dice",
            "Positive predicted empty (%)", "Empty-slice FP (%)",
        ],
        "Baseline": [
            baseline_best["val_global_micro_dice"],
            baseline_best["val_mean_patient_dice"],
            baseline_best["val_median_patient_dice"],
            baseline_best["val_positive_predicted_empty_pct"],
            baseline_best["val_empty_slice_false_positive_pct"],
        ],
        "Balanced sampler": [
            ablation_best["val_global_micro_dice"],
            ablation_best["val_mean_patient_dice"],
            ablation_best["val_median_patient_dice"],
            ablation_best["val_positive_predicted_empty_pct"],
            ablation_best["val_empty_slice_false_positive_pct"],
        ],
    })

    patient_comparison = (
        baseline_patients[["volume_id", "micro_dice"]]
        .rename(columns={"micro_dice": "Baseline"})
        .merge(
            best_patient_metrics[["volume_id", "micro_dice"]].rename(
                columns={"micro_dice": "Balanced sampler"}
            ),
            on="volume_id", how="inner",
        )
    )
    patient_comparison = patient_comparison.loc[
        patient_comparison["volume_id"].isin([104, 107, 108, 109, 110, 111, 112, 113, 116])
    ].sort_values("Baseline")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    axes[0].plot(
        baseline_history["epoch"], baseline_history["val_mean_patient_dice"],
        marker="o", color="#8A949E", label="Baseline",
    )
    axes[0].plot(
        history_frame["epoch"], history_frame["val_mean_patient_dice"],
        marker="s", color="#2878B5", label="Balanced sampler",
    )
    axes[0].axhline(
        baseline_best["val_mean_patient_dice"], linestyle="--",
        color="#333333", label="Baseline best",
    )
    axes[0].set_title("Mean patient Dice by epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylim(0, 1); axes[0].legend()

    x = np.arange(len(patient_comparison))
    width = 0.38
    axes[1].bar(
        x - width / 2, patient_comparison["Baseline"], width,
        color="#B9C2CC", edgecolor="#333333", label="Baseline",
    )
    axes[1].bar(
        x + width / 2, patient_comparison["Balanced sampler"], width,
        color="#2878B5", edgecolor="#333333", label="Balanced sampler",
    )
    axes[1].set_xticks(x, patient_comparison["volume_id"].astype(str))
    axes[1].set_title("Tumor-positive patient Dice")
    axes[1].set_xlabel("Validation volume"); axes[1].set_ylim(0, 1); axes[1].legend()

    delta = patient_comparison["Balanced sampler"] - patient_comparison["Baseline"]
    axes[2].barh(
        patient_comparison["volume_id"].astype(str), delta,
        color=np.where(delta >= 0, "#2878B5", "#E68632"),
        edgecolor="#333333",
    )
    axes[2].axvline(0, color="#333333", linewidth=1)
    axes[2].set_title("Patient Dice change versus baseline")
    axes[2].set_xlabel("Dice difference"); axes[2].set_ylabel("Validation volume")

    fig.suptitle("Controlled sampler ablation comparison", fontsize=17)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_comparison_dashboard.png", dpi=160, bbox_inches="tight")
    plt.show()
    display(metric_comparison.style.format({"Baseline": "{:.4f}", "Balanced sampler": "{:.4f}"}))
else:
    print("Comparison is available after training and best-checkpoint evaluation.")
'''),
]

notebook.cells[30:30] = comparison_cells

for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

notebook.metadata["experiment"] = {
    "name": "patient_lesion_balanced_sampler_ablation",
    "single_intervention": "training_sampler",
    "test_split_locked": True,
    "baseline_gate": str(
        ROOT / "Practice" / "patient_aware_baseline_outputs"
        / "patient_aware_gate_result.json"
    ),
}

nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")

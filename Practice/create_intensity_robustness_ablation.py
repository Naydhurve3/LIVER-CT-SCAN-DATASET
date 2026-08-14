"""Build the organ-normalized intensity-robustness training ablation notebook."""

from copy import deepcopy
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "auto_cool_continue_patient_aware_to_epoch10.ipynb"
OUTPUT = ROOT / "Practice" / "organ_normalized_intensity_robustness_ablation.ipynb"

notebook = deepcopy(nbformat.read(SOURCE, as_version=4))

notebook.cells[0].source = """# Organ-Normalized Intensity-Robustness Ablation

Volume 104 is an appearance outlier relative to the training distribution.
This controlled experiment returns to the strongest Focal-Dice baseline and
changes only image normalization and photometric augmentation.

The test split remains locked.
"""
notebook.cells[1].source = """## tl;dr

- Fresh 10-epoch model; do not resume any earlier experiment.
- Preserve Focal-Dice, `3×` positive-slice sampling, MobileNetV2-U-Net,
  optimizer, geometry augmentation and validation metrics.
- Normalize every slice using statistics inside its verified organ mask.
- Apply bounded gamma and Gaussian-noise augmentation only during training.
- Compare patient 104, patient 116, mean patient Dice and detection guardrails
  against the frozen Focal-Dice baseline.
"""
notebook.cells[2].source = """## Context & Methods

### Key assumptions

- The appearance-forensics notebook identified volume 104 as a multivariate
  appearance outlier and volume 116 as non-outlying.
- Organ masks are input preprocessing support, not the tumor target.
- The same deterministic normalization is applied to training and validation.
- Gamma/noise augmentation is training-only.
- Architecture, loss and sampler remain unchanged, isolating intensity handling.
"""

setup = notebook.cells[3].source
setup = setup.replace(
    "import pandas as pd\nimport torch",
    "import pandas as pd\nfrom PIL import Image\nimport torch",
)
setup = setup.replace(
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"',
    'BASELINE_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"\n'
    'COMPOSITE_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "stabilized_composite_loss_outputs"\n'
    'FORENSICS_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "appearance_domain_forensics_outputs"\n'
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "intensity_robustness_outputs"',
)
setup = setup.replace(
    "POSITIVE_SAMPLE_WEIGHT = 3.0",
    'PREPROCESSING_STRATEGY = "organ_robust_zscore_clip3_gamma_noise_v1"\n'
    "POSITIVE_SAMPLE_WEIGHT = 3.0\n"
    "ORGAN_Z_CLIP = 3.0\n"
    "GAMMA_RANGE = (0.85, 1.15)\n"
    "GAMMA_PROBABILITY = 0.50\n"
    "NOISE_STD_RANGE = (0.0, 0.025)\n"
    "NOISE_PROBABILITY = 0.35",
)
notebook.cells[3].source = setup

notebook.cells[8].source = "### 2. Define organ normalization and paired training augmentation"
notebook.cells[9].source = r'''from torch.utils.data import Dataset


def organ_robust_normalize(image: np.ndarray, organ_mask: np.ndarray) -> np.ndarray:
    """Normalize the full slice from robust statistics inside the organ mask."""
    image = np.asarray(image, dtype=np.float32)
    organ_mask = np.asarray(organ_mask, dtype=bool)
    reference = image[organ_mask]
    if reference.size < 32:
        reference = image[image > 0]
    if reference.size < 32:
        reference = image.reshape(-1)
    center = float(np.median(reference))
    q25, q75 = np.percentile(reference, [25, 75])
    robust_sigma = float((q75 - q25) / 1.349)
    if not np.isfinite(robust_sigma) or robust_sigma < 1e-3:
        robust_sigma = max(float(np.std(reference)), 1e-3)
    normalized = np.clip((image - center) / robust_sigma, -ORGAN_Z_CLIP, ORGAN_Z_CLIP)
    normalized = (normalized + ORGAN_Z_CLIP) / (2 * ORGAN_Z_CLIP)
    return normalized.astype(np.float32)


class PairedTrainingAugment:
    def __init__(
        self, flip_probability=0.30, affine_probability=0.60,
        max_rotation_degrees=10.0, max_translation_fraction=0.05,
        scale_range=(0.95, 1.05),
    ):
        self.flip_probability = flip_probability
        self.affine_probability = affine_probability
        self.max_rotation_degrees = max_rotation_degrees
        self.max_translation_fraction = max_translation_fraction
        self.scale_range = scale_range

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if random.random() < self.flip_probability:
            image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
        if random.random() < self.affine_probability:
            height, width = image.shape
            angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
            scale = random.uniform(*self.scale_range)
            tx = random.uniform(-self.max_translation_fraction, self.max_translation_fraction) * width
            ty = random.uniform(-self.max_translation_fraction, self.max_translation_fraction) * height
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            matrix[:, 2] += (tx, ty)
            image = cv2.warpAffine(
                image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            mask = cv2.warpAffine(
                mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        if random.random() < GAMMA_PROBABILITY:
            gamma = random.uniform(*GAMMA_RANGE)
            image = np.power(np.clip(image, 0, 1), gamma)
        if random.random() < NOISE_PROBABILITY:
            sigma = random.uniform(*NOISE_STD_RANGE)
            image = image + np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class OrganNormalizedDataset(Dataset):
    """Verified dataset wrapper using the row-matched verified organ mask."""
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        self.rows = base_dataset.rows

    def __len__(self):
        return len(self.base)

    @property
    def sample_ids(self):
        return self.base.sample_ids

    @property
    def tumor_positive_flags(self):
        return self.base.tumor_positive_flags

    def __getitem__(self, index):
        sample = self.base[index]
        image = sample["image"][0].numpy()
        mask = sample["mask"][0].numpy()
        with Image.open(self.rows[index]["organ_mask_path"]) as handle:
            organ = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
        image = organ_robust_normalize(image, organ)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        sample["image"] = torch.from_numpy(image[None]).float()
        sample["mask"] = torch.from_numpy(mask[None]).float()
        return sample


train_transform = PairedTrainingAugment()
'''

notebook.cells[10].source = "### 3. Build baseline-weighted loaders with organ normalization"
notebook.cells[11].source = r'''train_base_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="train", root_dir=DATASET_ROOT,
    target="tumor", transform=None, validate_paths=True,
)
val_base_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="val", root_dir=DATASET_ROOT,
    target="tumor", transform=None, validate_paths=True,
)
train_dataset = OrganNormalizedDataset(train_base_dataset, transform=train_transform)
val_dataset = OrganNormalizedDataset(val_base_dataset, transform=None)

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
preview_indices = list(iter(train_sampler))[:4096]
preview_rate = float(train_flags[preview_indices].mean())
sampling_audit = pd.DataFrame({
    "measure": ["Natural training", "Expected weighted", "Sampler preview"],
    "positive_slice_pct": [100 * natural_rate, 100 * expected_rate, 100 * preview_rate],
})
display(sampling_audit.style.format({"positive_slice_pct": "{:.2f}%"}))
print(f"Train={len(train_dataset):,} | Validation={len(val_dataset):,}")
'''

notebook.cells[12].source = "### 4. Visualize normalization and bounded intensity augmentation"
notebook.cells[13].source = r'''positive_indices = np.flatnonzero(train_flags)
preview_indices = np.random.default_rng(SEED).choice(
    positive_indices, size=min(4, len(positive_indices)), replace=False
)
fig, axes = plt.subplots(len(preview_indices), 3, figsize=(14, 4 * len(preview_indices)))
if len(preview_indices) == 1:
    axes = axes[None, :]
for row_axis, source_index in zip(axes, preview_indices):
    base_sample = train_base_dataset[int(source_index)]
    raw = base_sample["image"][0].numpy()
    mask = base_sample["mask"][0].numpy()
    with Image.open(train_base_dataset.rows[int(source_index)]["organ_mask_path"]) as handle:
        organ = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
    normalized = organ_robust_normalize(raw, organ)
    augmented, augmented_mask = train_transform(normalized.copy(), mask.copy())
    for axis, image, overlay, title in [
        (row_axis[0], raw, mask, "Frozen PNG input"),
        (row_axis[1], normalized, mask, "Organ-normalized"),
        (row_axis[2], augmented, augmented_mask, "Training augmentation"),
    ]:
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        if overlay.any():
            axis.contour(overlay, levels=[0.5], colors=["#E15759"], linewidths=0.8)
        axis.set_title(title); axis.axis("off")
fig.suptitle("Intensity-robustness preprocessing QA", fontsize=16)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "intensity_preprocessing_audit.png", dpi=160, bbox_inches="tight")
plt.show()
'''

for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace(
        '"positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,',
        '"positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,\n'
        '        "preprocessing_strategy": PREPROCESSING_STRATEGY,',
    )

notebook.cells[30].source = """## Takeaways

This is a single-intervention intensity-robustness ablation. Promote it only if
patient 104 recovers beyond the Focal-Dice baseline without losing patient 116,
mean patient Dice, or the detection guardrails.
"""
notebook.cells[31].source = r'''minimum_decision_epoch_reached = (
    not history_frame.empty
    and int(history_frame["epoch"].max()) >= MIN_EPOCHS_BEFORE_EARLY_STOP
)
baseline_gate = json.loads(
    (BASELINE_OUTPUT_DIR / "patient_aware_gate_result.json").read_text(encoding="utf-8")
)

if not minimum_decision_epoch_reached:
    completed_epochs = int(history_frame["epoch"].max()) if not history_frame.empty else 0
    final_gate = {
        "status": "incomplete_continue_to_epoch_10",
        "epochs_completed": completed_epochs,
        "minimum_decision_epoch": MIN_EPOCHS_BEFORE_EARLY_STOP,
        "preprocessing_strategy": PREPROCESSING_STRATEGY,
        "test_images_accessed": False,
        "decision": f"INCOMPLETE — continue from epoch {completed_epochs + 1}.",
    }
elif best_patient_metrics.empty or size_metrics.empty:
    final_gate = {
        "status": "not_run", "test_images_accessed": False,
        "decision": "Run training and best-checkpoint evaluation.",
    }
else:
    best_row = history_frame.loc[history_frame["val_mean_patient_dice"].idxmax()]
    positive = best_patient_metrics.loc[best_patient_metrics["true_pixels"].gt(0)]
    volume_104 = positive.loc[positive["volume_id"].eq(104), "micro_dice"]
    volume_116 = positive.loc[positive["volume_id"].eq(116), "micro_dice"]
    q1 = size_metrics.loc[size_metrics["size_quartile"].eq("Q1 smallest")].iloc[0]
    mean_improved = float(best_row["val_mean_patient_dice"]) > float(baseline_gate["best_mean_patient_dice"])
    volume_104_improved = bool(len(volume_104) and float(volume_104.iloc[0]) > float(baseline_gate["volume_104_dice"]))
    volume_116_acceptable = bool(len(volume_116) and float(volume_116.iloc[0]) > 0.05)
    q1_acceptable = float(q1["detected_pct"]) > 30.0
    guardrails = bool(
        float(best_row["val_positive_predicted_empty_pct"]) < 30.0
        and float(best_row["val_empty_slice_false_positive_pct"]) < 15.0
    )
    ready = all([mean_improved, volume_104_improved, volume_116_acceptable, q1_acceptable, guardrails])
    final_gate = {
        "status": "intensity_robustness_pass" if ready else "intensity_robustness_fail",
        "manifest_sha256": manifest_hash,
        "preprocessing_strategy": PREPROCESSING_STRATEGY,
        "epochs_completed": int(history_frame["epoch"].max()),
        "best_epoch": int(best_row["epoch"]),
        "best_global_micro_dice": float(best_row["val_global_micro_dice"]),
        "best_mean_patient_dice": float(best_row["val_mean_patient_dice"]),
        "baseline_mean_patient_dice": float(baseline_gate["best_mean_patient_dice"]),
        "volume_104_dice": float(volume_104.iloc[0]) if len(volume_104) else None,
        "baseline_volume_104_dice": float(baseline_gate["volume_104_dice"]),
        "volume_116_dice": float(volume_116.iloc[0]) if len(volume_116) else None,
        "q1_smallest_detected_pct": float(q1["detected_pct"]),
        "positive_predicted_empty_pct": float(best_row["val_positive_predicted_empty_pct"]),
        "empty_slice_false_positive_pct": float(best_row["val_empty_slice_false_positive_pct"]),
        "mean_patient_improved": mean_improved,
        "volume_104_improved": volume_104_improved,
        "volume_116_acceptable": volume_116_acceptable,
        "q1_detection_acceptable": q1_acceptable,
        "detection_guardrails_pass": guardrails,
        "test_images_accessed": False,
        "decision": (
            "PASS — retain organ normalization and confirm with a longer run."
            if ready else
            "FAIL — retain the original Focal-Dice baseline and test 2.5D context."
        ),
    }

(OUTPUT_DIR / "intensity_robustness_gate_result.json").write_text(
    json.dumps(final_gate, indent=2), encoding="utf-8"
)
display(pd.DataFrame([final_gate]).T.rename(columns={0: "result"}))
print(final_gate["decision"])
'''

comparison_cells = [
    new_markdown_cell("""### Baseline comparison dashboard

Compare the intensity-robust candidate against the frozen Focal-Dice baseline
at their respective best mean-patient-Dice checkpoints.
"""),
    new_code_cell(r'''baseline_history = pd.read_csv(
    BASELINE_OUTPUT_DIR / "patient_aware_history.csv"
)
baseline_patients = pd.read_csv(
    BASELINE_OUTPUT_DIR / "best_validation_patient_metrics.csv"
)
if not history_frame.empty and not best_patient_metrics.empty:
    baseline_best = baseline_history.loc[baseline_history["val_mean_patient_dice"].idxmax()]
    candidate_best = history_frame.loc[history_frame["val_mean_patient_dice"].idxmax()]
    patient_comparison = (
        baseline_patients[["volume_id", "micro_dice"]]
        .rename(columns={"micro_dice": "Baseline"})
        .merge(
            best_patient_metrics[["volume_id", "micro_dice"]]
            .rename(columns={"micro_dice": "Intensity robust"}),
            on="volume_id",
        )
    )
    patient_comparison = patient_comparison.loc[
        patient_comparison["volume_id"].isin([104,107,108,109,110,111,112,113,116])
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    axes[0].plot(baseline_history["epoch"], baseline_history["val_mean_patient_dice"], marker="o", color="#8A949E", label="Baseline")
    axes[0].plot(history_frame["epoch"], history_frame["val_mean_patient_dice"], marker="s", color="#2878B5", label="Intensity robust")
    axes[0].set_title("Mean patient Dice by epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylim(0,1); axes[0].legend()
    x=np.arange(len(patient_comparison)); width=0.38
    axes[1].bar(x-width/2,patient_comparison["Baseline"],width,color="#B9C2CC",edgecolor="#333333",label="Baseline")
    axes[1].bar(x+width/2,patient_comparison["Intensity robust"],width,color="#2878B5",edgecolor="#333333",label="Intensity robust")
    axes[1].set_xticks(x,patient_comparison["volume_id"].astype(str)); axes[1].set_ylim(0,1)
    axes[1].set_title("Tumor-positive patient Dice"); axes[1].legend()
    delta=patient_comparison["Intensity robust"]-patient_comparison["Baseline"]
    axes[2].barh(patient_comparison["volume_id"].astype(str),delta,color=np.where(delta>=0,"#2878B5","#E68632"),edgecolor="#333333")
    axes[2].axvline(0,color="#333333"); axes[2].set_title("Patient Dice change versus baseline"); axes[2].set_xlabel("Dice difference")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"intensity_baseline_comparison_dashboard.png",dpi=160,bbox_inches="tight")
    plt.show()
else:
    print("Comparison available after training.")
'''),
]
notebook.cells[30:30] = comparison_cells

for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

notebook.metadata["experiment"] = {
    "name": "organ_normalized_intensity_robustness_ablation",
    "single_intervention": "intensity_preprocessing",
    "test_split_locked": True,
}
nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")

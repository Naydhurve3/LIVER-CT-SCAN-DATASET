"""Build the adjacent-slice 2.5D context ablation notebook."""

from copy import deepcopy
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Practice" / "auto_cool_continue_patient_aware_to_epoch10.ipynb"
OUTPUT = ROOT / "Practice" / "adjacent_slice_2_5d_context_ablation.ipynb"

notebook = deepcopy(nbformat.read(SOURCE, as_version=4))

notebook.cells[0].source = """# Adjacent-Slice 2.5D Context Ablation

This controlled experiment restores the original frozen PNG intensity pipeline
and changes only the model input from one slice to three aligned channels:
previous, current and next slice.

The tumor target remains the current slice. The test split remains locked.
"""
notebook.cells[1].source = """## tl;dr

Organ normalization recovered volume 104 and small-lesion sensitivity, but
created 47.4% empty-slice false positives and reduced volume 116 to 0.0268.

The next hypothesis is that through-plane context can distinguish real tumor
continuity from isolated false positives. The notebook includes explicit
expected-versus-actual targets, progress bars and pass/fail graphs.
"""
notebook.cells[2].source = """## Context & Methods

### Key assumptions

- Original Focal-Dice loss, frozen PNG intensities and `3×` positive sampling
  are restored.
- Only the input context changes: `[slice-1, slice, slice+1]`.
- Volume boundaries repeat the nearest valid slice; channels never cross patients.
- Spatial and intensity augmentation parameters are shared across all channels.
- Validation and checkpoint selection remain patient-aware.
"""

setup = notebook.cells[3].source
setup = setup.replace(
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"',
    'BASELINE_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "patient_aware_baseline_outputs"\n'
    'INTENSITY_OUTPUT_DIR = PROJECT_ROOT / "Practice" / "intensity_robustness_outputs"\n'
    'OUTPUT_DIR = PROJECT_ROOT / "Practice" / "context_2_5d_outputs"',
)
setup = setup.replace(
    "POSITIVE_SAMPLE_WEIGHT = 3.0",
    'INPUT_STRATEGY = "adjacent_slice_previous_current_next_v1"\n'
    "POSITIVE_SAMPLE_WEIGHT = 3.0\n"
    "CONTEXT_OFFSETS = (-1, 0, 1)",
)
notebook.cells[3].source = setup

notebook.cells[8].source = "### 2. Define aligned three-slice loading and augmentation"
notebook.cells[9].source = r'''from torch.utils.data import Dataset


class PairedStackAugment:
    def __init__(
        self, flip_probability=0.30, affine_probability=0.60,
        max_rotation_degrees=10.0, max_translation_fraction=0.05,
        scale_range=(0.95, 1.05), intensity_probability=0.50,
        brightness_shift=0.05, contrast_range=(0.90, 1.10),
    ):
        self.flip_probability = flip_probability
        self.affine_probability = affine_probability
        self.max_rotation_degrees = max_rotation_degrees
        self.max_translation_fraction = max_translation_fraction
        self.scale_range = scale_range
        self.intensity_probability = intensity_probability
        self.brightness_shift = brightness_shift
        self.contrast_range = contrast_range

    def __call__(self, image_stack: np.ndarray, mask: np.ndarray):
        image_stack = np.asarray(image_stack, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if random.random() < self.flip_probability:
            image_stack = np.flip(image_stack, axis=2).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < self.affine_probability:
            _, height, width = image_stack.shape
            angle = random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
            scale = random.uniform(*self.scale_range)
            tx = random.uniform(-self.max_translation_fraction, self.max_translation_fraction) * width
            ty = random.uniform(-self.max_translation_fraction, self.max_translation_fraction) * height
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
            matrix[:, 2] += (tx, ty)
            image_stack = np.stack([
                cv2.warpAffine(
                    channel, matrix, (width, height), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                for channel in image_stack
            ])
            mask = cv2.warpAffine(
                mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        if random.random() < self.intensity_probability:
            contrast = random.uniform(*self.contrast_range)
            brightness = random.uniform(-self.brightness_shift, self.brightness_shift)
            image_stack = image_stack * contrast + brightness
        return (
            np.ascontiguousarray(np.clip(image_stack, 0, 1), dtype=np.float32),
            np.ascontiguousarray(mask > 0.5, dtype=np.float32),
        )


class AdjacentSliceDataset(Dataset):
    """Three image channels with the verified current-slice tumor target."""
    def __init__(self, base_dataset, transform=None):
        self.base = base_dataset
        self.transform = transform
        self.rows = base_dataset.rows
        self.lookup = {
            (int(row["volume_id"]), int(row["slice_index"])): index
            for index, row in enumerate(self.rows)
        }
        self.bounds = {}
        for row in self.rows:
            volume = int(row["volume_id"])
            self.bounds.setdefault(volume, []).append(int(row["slice_index"]))
        self.bounds = {
            volume: (min(indices), max(indices))
            for volume, indices in self.bounds.items()
        }

    def __len__(self):
        return len(self.base)

    @property
    def sample_ids(self):
        return self.base.sample_ids

    @property
    def tumor_positive_flags(self):
        return self.base.tumor_positive_flags

    def _image(self, index):
        with Image.open(self.rows[index]["image_path"]) as handle:
            return np.asarray(handle.convert("L"), dtype=np.float32) / 255.0

    def __getitem__(self, index):
        center = self.base[index]
        row = self.rows[index]
        volume, slice_index = int(row["volume_id"]), int(row["slice_index"])
        low, high = self.bounds[volume]
        channel_indices = [
            self.lookup[(volume, min(max(slice_index + offset, low), high))]
            for offset in CONTEXT_OFFSETS
        ]
        image_stack = np.stack([self._image(channel_index) for channel_index in channel_indices])
        mask = center["mask"][0].numpy()
        if self.transform is not None:
            image_stack, mask = self.transform(image_stack, mask)
        center["image"] = torch.from_numpy(image_stack).float()
        center["mask"] = torch.from_numpy(mask[None]).float()
        center["context_sample_ids"] = [self.rows[i]["sample_id"] for i in channel_indices]
        return center


train_transform = PairedStackAugment()
'''

# Ensure PIL is imported.
setup = notebook.cells[3].source
setup = setup.replace("import pandas as pd\nimport torch", "import pandas as pd\nfrom PIL import Image\nimport torch")
notebook.cells[3].source = setup

notebook.cells[10].source = "### 3. Build baseline-weighted 2.5D loaders"
notebook.cells[11].source = r'''train_base_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="train", root_dir=DATASET_ROOT,
    target="tumor", transform=None, validate_paths=True,
)
val_base_dataset = VerifiedManifestDataset(
    MANIFEST_PATH, split="val", root_dir=DATASET_ROOT,
    target="tumor", transform=None, validate_paths=True,
)
train_dataset = AdjacentSliceDataset(train_base_dataset, transform=train_transform)
val_dataset = AdjacentSliceDataset(val_base_dataset, transform=None)

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
expected_rate = POSITIVE_SAMPLE_WEIGHT * natural_rate / (
    POSITIVE_SAMPLE_WEIGHT * natural_rate + 1 - natural_rate
)
preview_indices = list(iter(train_sampler))[:4096]
preview_rate = float(train_flags[preview_indices].mean())
sampling_audit = pd.DataFrame({
    "measure": ["Natural training", "Expected weighted", "Sampler preview"],
    "positive_slice_pct": [100*natural_rate,100*expected_rate,100*preview_rate],
})
display(sampling_audit.style.format({"positive_slice_pct":"{:.2f}%"}))
print(f"Train={len(train_dataset):,} | Validation={len(val_dataset):,}")
'''

notebook.cells[12].source = "### 4. Visualize adjacent-slice alignment and expected targets"
notebook.cells[13].source = r'''positive_indices = np.flatnonzero(train_flags)
preview_indices = np.random.default_rng(SEED).choice(
    positive_indices, size=min(3, len(positive_indices)), replace=False
)
fig, axes = plt.subplots(len(preview_indices), 4, figsize=(16, 4*len(preview_indices)))
if len(preview_indices)==1:
    axes=axes[None,:]
for row_axes,index in zip(axes,preview_indices):
    sample=val_dataset[int(index % len(val_dataset))]
    mask=sample["mask"][0].numpy()
    for channel,axis,title in zip(sample["image"],row_axes[:3],["Previous","Current","Next"]):
        axis.imshow(channel,cmap="gray",vmin=0,vmax=1)
        axis.set_title(title); axis.axis("off")
    row_axes[3].imshow(sample["image"][1],cmap="gray",vmin=0,vmax=1)
    if mask.any(): row_axes[3].contour(mask,levels=[0.5],colors=["#E15759"],linewidths=1)
    row_axes[3].set_title("Current target"); row_axes[3].axis("off")
fig.suptitle("2.5D channel alignment QA",fontsize=16); fig.tight_layout()
fig.savefig(OUTPUT_DIR/"context_alignment_audit.png",dpi=160,bbox_inches="tight")
plt.show()

expected_targets = pd.DataFrame([
    {"metric":"Mean patient Dice","direction":"higher","target":0.383069,"baseline":0.383069},
    {"metric":"Volume 104 Dice","direction":"higher","target":0.197613,"baseline":0.197613},
    {"metric":"Volume 116 Dice","direction":"higher","target":0.050000,"baseline":0.000000},
    {"metric":"Q1 detection (%)","direction":"higher","target":35.000000,"baseline":25.475285},
    {"metric":"Positive predicted empty (%)","direction":"lower","target":30.000000,"baseline":39.443378},
    {"metric":"Empty-slice FP (%)","direction":"lower","target":15.000000,"baseline":3.349580},
])
expected_targets.to_csv(OUTPUT_DIR/"expected_targets.csv",index=False)
display(expected_targets)
'''

for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace(
        "in_channels=1, out_channels=1",
        "in_channels=3, out_channels=1",
    )
    cell.source = cell.source.replace(
        '"positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,',
        '"positive_sample_weight": POSITIVE_SAMPLE_WEIGHT,\n'
        '        "input_strategy": INPUT_STRATEGY,',
    )
    cell.source = cell.source.replace(
        """review_dataset = VerifiedManifestDataset(
        MANIFEST_PATH, split="val", root_dir=DATASET_ROOT,
        target="tumor", sample_ids=review_ids,
    )
    review_loader = DataLoader(review_dataset, batch_size=4, shuffle=False, num_workers=0)""",
        """review_indices = [
        val_base_dataset.sample_ids.index(sample_id) for sample_id in review_ids
    ]
    review_dataset = torch.utils.data.Subset(val_dataset, review_indices)
    review_loader = DataLoader(review_dataset, batch_size=4, shuffle=False, num_workers=0)""",
    )
    cell.source = cell.source.replace(
        'image = item["image"][0].numpy()\n        truth = item["mask"][0].numpy().astype(bool)',
        'image = item["image"][1].numpy()\n        truth = item["mask"][0].numpy().astype(bool)',
    )

notebook.cells[30].source = """## Takeaways

The expected-versus-actual dashboard is the primary decision surface. The
candidate must improve patient robustness while remaining below both error-rate
ceilings; global Dice alone cannot pass the experiment.
"""

notebook.cells[31].source = r'''minimum_decision_epoch_reached = (
    not history_frame.empty and int(history_frame["epoch"].max()) >= MIN_EPOCHS_BEFORE_EARLY_STOP
)
baseline_gate = json.loads((BASELINE_OUTPUT_DIR/"patient_aware_gate_result.json").read_text(encoding="utf-8"))
if not minimum_decision_epoch_reached:
    completed=int(history_frame["epoch"].max()) if not history_frame.empty else 0
    final_gate={"status":"incomplete_continue_to_epoch_10","epochs_completed":completed,
        "input_strategy":INPUT_STRATEGY,"test_images_accessed":False,
        "decision":f"INCOMPLETE — continue from epoch {completed+1}."}
elif best_patient_metrics.empty or size_metrics.empty:
    final_gate={"status":"not_run","test_images_accessed":False,"decision":"Run training and evaluation."}
else:
    best_row=history_frame.loc[history_frame["val_mean_patient_dice"].idxmax()]
    positive=best_patient_metrics.loc[best_patient_metrics["true_pixels"].gt(0)]
    v104=float(positive.loc[positive["volume_id"].eq(104),"micro_dice"].iloc[0])
    v116=float(positive.loc[positive["volume_id"].eq(116),"micro_dice"].iloc[0])
    q1=float(size_metrics.loc[size_metrics["size_quartile"].eq("Q1 smallest"),"detected_pct"].iloc[0])
    actuals={
        "Mean patient Dice":float(best_row["val_mean_patient_dice"]),
        "Volume 104 Dice":v104,"Volume 116 Dice":v116,"Q1 detection (%)":q1,
        "Positive predicted empty (%)":float(best_row["val_positive_predicted_empty_pct"]),
        "Empty-slice FP (%)":float(best_row["val_empty_slice_false_positive_pct"]),
    }
    outcome=expected_targets.copy()
    outcome["actual"]=outcome["metric"].map(actuals)
    outcome["passed"]=np.where(
        outcome["direction"].eq("higher"),outcome["actual"]>outcome["target"],
        outcome["actual"]<outcome["target"],
    )
    outcome["target_progress"]=np.where(
        outcome["direction"].eq("higher"),
        outcome["actual"]/outcome["target"],
        outcome["target"]/outcome["actual"].clip(lower=1e-9),
    )
    outcome.to_csv(OUTPUT_DIR/"expected_vs_actual_results.csv",index=False)
    ready=bool(outcome["passed"].all())
    final_gate={
        "status":"context_2_5d_pass" if ready else "context_2_5d_fail",
        "manifest_sha256":manifest_hash,"input_strategy":INPUT_STRATEGY,
        "epochs_completed":int(history_frame["epoch"].max()),"best_epoch":int(best_row["epoch"]),
        "best_global_micro_dice":float(best_row["val_global_micro_dice"]),
        "best_mean_patient_dice":actuals["Mean patient Dice"],"volume_104_dice":v104,
        "volume_116_dice":v116,"q1_smallest_detected_pct":q1,
        "positive_predicted_empty_pct":actuals["Positive predicted empty (%)"],
        "empty_slice_false_positive_pct":actuals["Empty-slice FP (%)"],
        "all_expected_targets_passed":ready,"test_images_accessed":False,
        "decision":"PASS — confirm 2.5D context longer." if ready else "FAIL — retain baseline and review the expected-versus-actual gaps.",
    }
(OUTPUT_DIR/"context_2_5d_gate_result.json").write_text(json.dumps(final_gate,indent=2),encoding="utf-8")
display(pd.DataFrame([final_gate]).T.rename(columns={0:"result"}))
print(final_gate["decision"])
'''

expected_cells = [
    new_markdown_cell("""### Expected versus actual results dashboard

This section displays exactly what was required, what the model achieved, the
absolute gap, normalized target progress, and pass/fail status.
"""),
    new_code_cell(r'''if "outcome" in globals():
    outcome["gap"] = np.where(
        outcome["direction"].eq("higher"),
        outcome["actual"]-outcome["target"],
        outcome["target"]-outcome["actual"],
    )
    fig,axes=plt.subplots(2,2,figsize=(18,12))
    y=np.arange(len(outcome))
    axes[0,0].barh(y,outcome["baseline"],color="#B9C2CC",edgecolor="#333333",label="Previous baseline")
    axes[0,0].scatter(outcome["target"],y,color="#E68632",marker="D",s=70,label="Required")
    axes[0,0].scatter(outcome["actual"],y,color="#2878B5",marker="o",s=70,label="Actual")
    axes[0,0].set_yticks(y,outcome["metric"]); axes[0,0].set_title("Expected, baseline, and actual values"); axes[0,0].legend()
    colors=np.where(outcome["passed"],"#2878B5","#E68632")
    axes[0,1].barh(outcome["metric"],outcome["target_progress"],color=colors,edgecolor="#333333")
    axes[0,1].axvline(1,color="#333333",linestyle="--"); axes[0,1].set_title("Target progress ratio"); axes[0,1].set_xlabel("1.0 means target reached")
    axes[1,0].barh(outcome["metric"],outcome["gap"],color=colors,edgecolor="#333333")
    axes[1,0].axvline(0,color="#333333"); axes[1,0].set_title("Signed pass margin"); axes[1,0].set_xlabel("Positive means target passed")
    axes[1,1].axis("off")
    table_data=outcome[["metric","target","actual","passed"]].copy()
    table_data["target"]=table_data["target"].map(lambda x:f"{x:.4f}")
    table_data["actual"]=table_data["actual"].map(lambda x:f"{x:.4f}")
    axes[1,1].table(cellText=table_data.values,colLabels=table_data.columns,loc="center",cellLoc="center")
    axes[1,1].set_title("Required outputs and observed results")
    fig.suptitle("2.5D context expected-versus-actual dashboard",fontsize=17)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/"expected_vs_actual_dashboard.png",dpi=170,bbox_inches="tight")
    plt.show()
    display(outcome.style.format({"target":"{:.4f}","baseline":"{:.4f}","actual":"{:.4f}","gap":"{:+.4f}","target_progress":"{:.2f}"}))
else:
    print("Expected-versus-actual dashboard is available after epoch-10 evaluation.")
'''),
]
notebook.cells[32:32] = expected_cells

for cell in notebook.cells:
    if cell.cell_type == "code":
        cell.execution_count = None
        cell.outputs = []

notebook.metadata["experiment"]={"name":"adjacent_slice_2_5d_context_ablation","single_intervention":"input_context","test_split_locked":True}
nbformat.validate(notebook)
nbformat.write(notebook,OUTPUT)
print(f"Wrote {OUTPUT}")

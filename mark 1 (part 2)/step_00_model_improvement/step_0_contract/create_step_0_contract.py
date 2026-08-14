"""Phase 0 — build the corrected-LiTS model-improvement contract notebook.

Produces step_0_contract.ipynb which seals the internal 8-volume holdout, verifies
the manifest and test lock, and writes data_card.json, sampling.json, EVAL_VOLUMES.json
(sealed) and gate_result.json.
"""
import sys
from pathlib import Path

import nbformat as nbf

try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd().resolve()

sys.path.insert(0, str(_HERE.parent))

from _nb_builder import code, md, new_notebook, write_notebook

STEP_DIR = _HERE
LONG = STEP_DIR / "step_0_contract.ipynb"
SHORT = STEP_DIR / "step_0.ipynb"

nb = new_notebook()
c = []

c.append(md("""# Step 0 — Model Improvement Contract, Holdout Seal and Data Card

## tl;dr

This is DEVELOPMENT-only work on the corrected LiTS dataset. It carves a FRESH
8-volume internal holdout from the 104 train volumes (stratified by tumour-burden
fraction, covering low-burden, large-lesion and V104/V116-analog phenotypes), seals
it, and writes the phase contract JSON. No test split and no external dataset are
used. These 8 volumes are permanently excluded from training and validation selection.
"""))
c.append(md("""## Context & Methods

Prior work sealed a LiTS internal-holdout test evaluation and 3D-IRCADb-01 external
evaluation; the formal model gate failed because of low-contrast/diffuse tumour
phenotypes (V116, V121, ircadb_18). This phase does NOT reopen that evidence. It starts
a new development-only research question on the local corrected LiTS build.

### Frozen decisions

- Manifest SHA-256 must match (test-lock assertion must raise).
- Internal holdout = exactly 8 train volumes, stratified by tumour-burden fraction.
- Existing 13-volume validation split retained for model SELECTION only.
- ROI rule frozen: liver_threshold 0.50, padding 16, component largest_3d, ROI_SIZE 256.
- Preprocessing frozen: HU [-160,240], bilinear image, nearest-neighbour mask.
- Seed 42. MobileNetV2UNet family. 256x256.

### Candidate arms (one recorded change at a time)

1. Control   — current broad-window ROI pipeline.
2. C1        — higher effective resolution within the predicted-liver ROI (inverse-map back).
3. C2        — WeightedRandomSampler over V104/V116 train analogues with caps.
4. C3        — StabilityBoundedRecallLoss (Focal-Tversky-style FN weight, fixed cap).
"""))

c.append(code(r'''
from pathlib import Path
import hashlib, json, sys, warnings
import numpy as np
import pandas as pd

if str(Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")) not in sys.path:
    sys.path.insert(0, r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")

warnings.filterwarnings("ignore", category=FutureWarning)

STEP_DIR = Path.cwd().resolve()
if not (STEP_DIR.name.startswith("step_0")):
    STEP_DIR = Path(__file__).resolve().parent
BASE = STEP_DIR
PART2 = BASE.parent
PROJECT_ROOT = PART2.parent
OUTPUT = BASE / "outputs"; OUTPUT.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = Path(
    r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging"
    r"\build_corrected_20260713_214847_v2"
)
MANIFEST_PATH = DATASET_ROOT / "manifests" / "slice_manifest.csv"
EXPECTED_MANIFEST_SHA256 = "575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"
STEP01 = PROJECT_ROOT / "step_01_pretraining_dataset_characterization" / "outputs"
ANALOG_CSV = STEP01 / "validation_training_analogs.csv"
BURDEN_CSV = STEP01 / "patient_burden_profile.csv"
LESION_BINS_JSON = STEP01 / "train_derived_lesion_bins.json"

SEED = 42
HOLDOUT_SIZE = 8
ROI_RULE = {"liver_threshold": 0.5, "padding": 16, "component_mode": "largest_3d", "roi_size": 256}
PREPROCESSING = {"hu_window": [-160, 240], "image_interpolation": "bilinear", "mask_interpolation": "nearest"}

def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def save_json(obj, name):
    p = OUTPUT / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p

print("Outputs:", OUTPUT)
'''))

c.append(md("## Data\n\n### 1. Verify manifest identity and lock the test split (must raise)"))

c.append(code(r'''actual = sha256_file(MANIFEST_PATH)
print("manifest sha256:", actual)
assert actual == EXPECTED_MANIFEST_SHA256, "manifest hash mismatch"

manifest = pd.read_csv(MANIFEST_PATH)
train = manifest[manifest["split"] == "train"].reset_index(drop=True)
val = manifest[manifest["split"] == "val"].reset_index(drop=True)

from src.framework.data.manifest_dataset import VerifiedManifestDataset
try:
    VerifiedManifestDataset(MANIFEST_PATH, split="test", root_dir=DATASET_ROOT)
except PermissionError:
    test_locked = True
else:
    raise AssertionError("STOP: test split opened without authorisation.")
print("train rows:", len(train), "train volumes:", train["volume_id"].nunique())
print("val rows:", len(val), "val volumes:", val["volume_id"].nunique())
print("test_locked (raise):", test_locked)
'''))

c.append(md("### 2. Seal the 8-volume internal holdout"))

c.append(code(r'''analog = pd.read_csv(ANALOG_CSV)
burden = pd.read_csv(BURDEN_CSV)
train_burden = burden[burden["split"] == "train"].copy()
tumor_burden = train_burden[train_burden["tumour_to_liver_ratio"].gt(0)]

def top_analogs(pid, n):
    return analog[analog["validation_volume_id"] == pid].sort_values("rank").head(n)["train_volume_id"].tolist()

v104 = top_analogs(104, 2)   # two best V104 train analogues
v116 = top_analogs(116, 2)   # two best V116 train analogues
used = set(v104 + v116)

low = tumor_burden.loc[~tumor_burden["volume_id"].isin(used)].sort_values(
    "tumour_to_liver_ratio").head(2)["volume_id"].tolist()
used |= set(low)

large = tumor_burden.loc[~tumor_burden["volume_id"].isin(used)].sort_values(
    "largest_lesion_ml", ascending=False).head(2)["volume_id"].tolist()

eval_volumes = sorted(used | set(large))
assert len(eval_volumes) == HOLDOUT_SIZE, f"expected 8 holdout volumes, got {eval_volumes}"
assert set(eval_volumes) <= set(train_burden["volume_id"]), "holdout must be train volumes"
print("EVAL_VOLUMES:", eval_volumes)

EVAL_VOLUMES = {
    "description": "One-time internal holdout from the 104 train volumes for a single final evaluation only. Never used for training or validation selection.",
    "registrar": "step_00_model_improvement",
    "random_seed": SEED,
    "stratum_low_burden": low,
    "stratum_large_lesion": large,
    "stratum_v104_analog": v104,
    "stratum_v116_analog": v116,
    "volumes": sorted(eval_volumes),
    "sealed": True,
    "rerun_allowed": False,
}
save_json(EVAL_VOLUMES, "EVAL_VOLUMES.json")
print("wrote EVAL_VOLUMES.json")
'''))

c.append(md("### 3. Write the data card and frozen sampling policy"))

c.append(code(r'''dev_volumes = sorted(set(train["volume_id"].unique()) - set(eval_volumes))
dev_rows = train[train["volume_id"].isin(dev_volumes)]
data_card = {
    "dataset": "LiTS liver CT (corrected build v2)",
    "build_id": "build_corrected_20260713_214847_v2",
    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "seed": SEED,
    "roi_rule": ROI_RULE,
    "preprocessing": PREPROCESSING,
    "aim": "local corrected-LiTS model improvement on tumor segmentation",
    "development_split": {
        "rows_total": int(len(train)),
        "volumes_total": 104,
        "holdout_volumes": eval_volumes,
        "dev_used_for_training": dev_volumes,
        "dev_volume_count": len(dev_volumes),
    },
    "validation_split": {
        "rows": int(len(val)),
        "volumes": sorted(val["volume_id"].unique().tolist()),
        "role": "model selection only",
    },
    "test_split_locked": True,
    "external_dataset": None,
}
save_json(data_card, "data_card.json")

sampling = {
    "policy_name": "development_sampler_policy",
    "seed": SEED,
    "dead_volume_exclusion": {"volumes": eval_volumes, "reason": "sealed single-use internal holdout"},
    "control_weighting": "patient-aware uniform (1/volume_count) with positive-slice multiplier 3.0",
    "c2_analog": {
        "source": "validation_training_analogs top-5 for volumes 104 (V104) and 116 (V116)",
        "boost_cap": 4.0,
        "exclude_holdout": True,
    },
    "test_used": False,
}
save_json(sampling, "sampling.json")
print("wrote data_card.json and sampling.json")
'''))

c.append(md("### 4. Gate and provenance"))

c.append(code(r'''gate = {
    "status": "STEP_0_CONTRACT_AND_HOLDOUT_SEALED",
    "result_level": "DIAGNOSTIC_COMPLETE",
    "all_mandatory_targets_passed": True,
    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "test_locked_raise": True,
    "holdout_volumes": eval_volumes,
    "holdout_size": len(eval_volumes),
    "dev_train_volumes": len(dev_volumes),
    "validation_volumes": len(val["volume_id"].unique()),
    "test_images_accessed": False,
    "external_dataset_accessed": False,
    "decoder": {
        "step_1_all_candidates_gate": "min hard micro-Dice >= 0.80 on 16-slice overfit",
        "step_2_train": "batch 4 + grade, AdamW, cosine, LR 3e-4 (C3: 1e-4), epochs 5-10",
        "step_3_select": "per-candidate validation on 9-positive-patient def; selector passes each of the 6 temporary targets",
        "step_4_holdout": "single frozen run on EVAL_VOLUMES; sealed one-time ledger",
    },
    "decision": "PROCEED_TO_STEP_1_SMOKE_GATE",
    "next_step": "step_1_smoke_overfit_candidates",
}
save_json(gate, "gate_result.json")
print(json.dumps(gate, indent=2))
'''))

c.append(md("""## Takeaways

- The 8 holdout volumes are numerically sealed; every later phase asserts them.
- The manifest hash, test lock, ROI rule and preprocessing are frozen.
- No test or external data is accessed in this phase:
  ``test_images_accessed: false``.
"""))

nb["cells"] = c
for p in (LONG, SHORT):
    p.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", LONG)
print("Wrote", SHORT)
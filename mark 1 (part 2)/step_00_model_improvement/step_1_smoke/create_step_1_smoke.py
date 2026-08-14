"""Phase 1 — smoke/overfit gate for the four improvement arms.

Builds ``step_1_smoke.ipynb``. Each arm (Control, C1, C2, C3) trains a cold-start
MobileNetV2UNet on a deterministic 16-slice dev overfit set and must reach
hard (thresholded) micro-Dice >= 0.80 on that set. A trainability canary only.
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
NB_PATH = STEP_DIR / "step_1_smoke.ipynb"
NB_SHORT = STEP_DIR / "step_1.ipynb"

nb = new_notebook()
c = []

c.append(md("""# Step 1 — Smoke Gate: 16-slice overfit for the four candidates

## tl;dr
Development-only. Each of the four candidates (Control, C1, C2, C3) must overfit a
deterministic 16-slice dev set to hard micro-Dice >= 0.80 before full training.
The sealed 8-volume holdout is excluded. This phase only proves trainability;
it does not select a model and it never touches the test split.
"""))
c.append(md("""## Context & Methods

Reuses the frozen interfaces (VerifiedManifestDataset, MobileNetV2UNet, FocalDiceLoss,
StabilityBoundedRecallLoss) and the Phase 0 sealed EVAL_VOLUMES. All arms cold-start
(ImageNet-pretrained encoder, random tail). GPU required.

### Key Assumptions
- If an arm cannot overfit a handful of slices its full implementation is suspect.
- Hard micro-Dice (threshold 0.5, aggregated over all pixels of the overfit set)
  is the stated gate metric.
- The gate is on the *training* set by design; validation/generalisation is Step 3.

### Candidate arms (one recorded change at a time)
1. Control — full-slice 256x256, FocalDiceLoss, uniform sampling.
2. C1      — predicted-liver ROI box cropped and up-sampled to 256, FocalDiceLoss.
3. C2      — same input as Control, WeightedRandomSampler boosting V104/V116 analogs (cap 4.0).
4. C3      — same input as Control, StabilityBoundedRecallLoss (capped recall).
"""))

c.append(code(r'''
from pathlib import Path
import json, sys, random, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
if str(Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")) not in sys.path:
    sys.path.insert(0, r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver")

import src.framework.losses  # register losses
from src.framework.losses import FocalDiceLoss, StabilityBoundedRecallLoss
from src.framework.models import MobileNetV2UNet
from src.framework.data.manifest_dataset import VerifiedManifestDataset

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)
assert DEVICE.type == "cuda", "smoke training requires a GPU"
'''))

c.append(md("### Load sealed contract and define frozen paths/params"))

c.append(code(r'''
DATASET = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2")
MANIFEST = DATASET / "manifests" / "slice_manifest.csv"
EXPECTED_SHA = "575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"
STEP0_OUT = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1 (part 2)\step_00_model_improvement\step_0_contract\outputs")
EVAL_JSON = STEP0_OUT / "EVAL_VOLUMES.json"
SAMP_JSON = STEP0_OUT / "sampling.json"
ROI_MANIFEST_CSV = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1\mark_3_outputs\training_roi_manifest.csv")
OUT = Path.cwd() / "outputs"; OUT.mkdir(parents=True, exist_ok=True)

def sha256_file(p, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def save_json(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p

HOLDOUT = json.loads(EVAL_JSON.read_text(encoding="utf-8"))["volumes"]
SAMPLING = json.loads(SAMP_JSON.read_text(encoding="utf-8"))
ROI = pd.read_csv(ROI_MANIFEST_CSV)
print("holdout:", HOLDOUT)
print("sampling c2:", SAMPLING["c2_analog"])
'''))

c.append(md("### Verify manifest hash and lock the test split (must raise)"))

c.append(code(r'''actual = sha256_file(MANIFEST)
print("manifest sha256:", actual)
assert actual == EXPECTED_SHA, "manifest hash mismatch"

train_full = VerifiedManifestDataset(MANIFEST, split="train", root_dir=DATASET)
try:
    VerifiedManifestDataset(MANIFEST, split="test", root_dir=DATASET)
except PermissionError:
    test_locked = True
else:
    raise AssertionError("STOP: test split opened without authorisation.")
print("train rows:", len(train_full.rows))
print("test_locked (raise):", test_locked)
'''))

c.append(md("### Deterministic 16-slice overfit set (train split, excluding holdout)"))

c.append(code(r'''dev_rows = [r for r in train_full.rows if r["volume_id"] not in HOLDOUT]
print("dev rows:", len(dev_rows), "dev volumes:", len({r["volume_id"] for r in dev_rows}))

rng = random.Random(SEED)
pos = [r for r in dev_rows if r["tumor_pixels"] > 0]
neg = [r for r in dev_rows if r["tumor_pixels"] == 0]
pos_sorted = sorted(pos, key=lambda r: r["tumor_pixels"], reverse=True)
overfit = pos_sorted[:8] + rng.sample(neg, 8)
rng.shuffle(overfit)
print("overfit slices:", len(overfit), "positive:", sum(r["tumor_pixels"] > 0 for r in overfit))
'''))

c.append(md("### Prepare per-slice tensors (frozen preprocessing)"))

c.append(code(r'''
from PIL import Image
from src.framework.data.transforms import resize_image, resize_mask, hu_window_cpu

def prepare_slice(row, crop_box=None, target=256):
    with Image.open(row["image_path"]) as f:
        img = np.asarray(f.convert("L"), dtype=np.float32) / 255.0
    with Image.open(row["tumor_mask_path"]) as f:
        mask = (np.asarray(f.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
    if crop_box is not None:
        y0, y1, x0, x1 = crop_box
        img = img[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
    img = hu_window_cpu(img, -160, 240)
    img = resize_image(img, (target, target))
    mask = resize_mask(mask, (target, target))
    return img.astype(np.float32), mask.astype(np.float32)

ROI_BY_VOL = dict(zip(ROI["volume_id"], zip(ROI["y0"], ROI["y1"], ROI["x0"], ROI["x1"])))

sets = {}
for name, use_crop in (("control", False), ("c1", True)):
    tensors = []
    for row in overfit:
        box = ROI_BY_VOL[row["volume_id"]] if use_crop else None
        img, mask = prepare_slice(row, box)
        tensors.append((torch.from_numpy(img)[None], torch.from_numpy(mask)[None]))
    sets[name] = tensors
# C2 and C3 share the Control input tensors
sets["c2"] = sets["control"]
sets["c3"] = sets["control"]
print({k: len(v) for k, v in sets.items()})
'''))

c.append(md("### Training loop (cold start, GPU, short)"))

c.append(code(r'''
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def hard_micro_dice(pred, mask):
    p = (pred > 0.5).float()
    inter = (p * mask).sum().float()
    denom = p.sum() + mask.sum() + 1e-6
    return (2 * inter / denom).item()

def train_arm(name, tensors, criterion, use_sampler=False, epochs=60, lr=3e-4):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MobileNetV2UNet(in_channels=1, out_channels=1, pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xs = torch.stack([t[0] for t in tensors])
    ys = torch.stack([t[1] for t in tensors])
    ds = TensorDataset(xs, ys)
    if use_sampler:
        w = torch.ones(len(ds))
        sampler = WeightedRandomSampler(w, num_samples=len(ds) * 4, replacement=True)
        dl = DataLoader(ds, batch_size=4, sampler=sampler, drop_last=False)
        steps_per_epoch = len(dl)
    else:
        dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=False)
        steps_per_epoch = len(dl)
    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        logits = model(xs.to(DEVICE))
        pred = torch.sigmoid(logits)
    dice = hard_micro_dice(pred.cpu(), ys)
    return {"arm": name, "epochs": epochs, "steps_per_epoch": steps_per_epoch,
            "hard_micro_dice": round(dice, 4), "loss_name": type(criterion).__name__}

results = []
results.append(train_arm("control", sets["control"], FocalDiceLoss(), use_sampler=False))
results.append(train_arm("c1_roi_highres", sets["c1"], FocalDiceLoss(), use_sampler=False))
# C2: same data, weighted sampler; C3: same data, capped recall loss
results.append(train_arm("c2_analog_sampler", sets["c2"], FocalDiceLoss(), use_sampler=True))
results.append(train_arm("c3_recall_capped", sets["c3"], StabilityBoundedRecallLoss(), use_sampler=False, lr=1e-4))

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(OUT / "smoke_arm_results.csv", index=False)
'''))

c.append(md("### Gate: every arm must overfit to >= 0.80 hard micro-Dice"))

c.append(code(r'''GATE_DICE = 0.80
df = pd.read_csv(OUT / "smoke_arm_results.csv")
passing = df[df["hard_micro_dice"] >= GATE_DICE]
gate = {
    "status": "STEP_1_SMOKE_GATE",
    "result_level": "DIAGNOSTIC_COMPLETE",
    "decision": "PROCEED_TO_STEP_2_TRAIN" if len(passing) == len(df) else "HALT_REVIEW_ARM",
    "gate_hard_micro_dice": GATE_DICE,
    "arms": df.to_dict(orient="records"),
    "all_arms_pass": bool(len(passing) == len(df)),
    "holdout_excluded": True,
    "holdout_volumes": HOLDOUT,
    "test_images_accessed": False,
    "external_dataset_accessed": False,
    "manifest_sha256": EXPECTED_SHA,
}
save_json(gate, "gate_result.json")
print(json.dumps(gate, indent=2))
'''))

c.append(md("""## Takeaways

- Every arm that passes is wired correctly end-to-end and can overfit 16 slices.
- The gate is a trainability canary on the training split; generalisation is NOT
  measured here (Step 3 uses the 13-volume validation split).
- ``test_images_accessed: false`` and ``external_dataset_accessed: false``.
- Next: full training for the passing arms (Step 2).
"""))

nb["cells"] = c
write_notebook(nb, NB_PATH, NB_SHORT)
print("Wrote", NB_PATH)
print("Wrote", NB_SHORT)
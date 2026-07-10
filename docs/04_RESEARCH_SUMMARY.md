# UP³RE-Net: Research Training Summary

> **Uncertainty-Propagated Per-Pixel Reweighting Ensemble Network**
> Binary liver tumor segmentation from 2D CT slices | RTX 3050 Ti 4GB laptop GPU

---

## 1. Problem Statement

Liver tumor segmentation from CT scans faces two critical challenges:

1. **Extreme class imbalance**: ~937:1 background-to-tumor ratio — standard losses (BCE, Dice) under-perform without re-weighting
2. **High uncertainty regions**: tumor boundaries, small lesions, ambiguous tissue — single models are overconfident at error locations

Existing solutions address these separately. **No prior work combines ensemble uncertainty with per-pixel loss re-weighting in a feedback loop** — this is the patent gap (Claims 1-6 in `docs/RESEARCH_NOVELTY.md`).

---

## 2. Architecture Overview

### 2.1 MobileNetV2U-Net (Base Model)

```
Input: (B, 1, 256, 256)
    │
    ▼
MobileNetV2 Encoder (torchvision, pretrained on ImageNet)
├── enc_0: Conv(1→32, k=3, s=2) — adapted from RGB→grayscale by weight averaging
├── enc_1: features[1:3]  →  C=24,  stride=2
├── enc_2: features[3:5]  →  C=32,  stride=4
├── enc_3: features[5:8]  →  C=64,  stride=8
├── enc_4: features[8:15] →  C=160, stride=16
└── enc_5: features[15:]  →  C=1280, stride=32
    │
    ▼
UNetDecoder (from scratch)
├── Upsample (scale=2) + ConvBlock(1280+160 → 256)
├── Upsample (scale=2) + ConvBlock(256+64 → 128)
├── Upsample (scale=2) + ConvBlock(128+32 → 64)
├── Upsample (scale=2) + ConvBlock(64+24 → 32)
├── Upsample (scale=2) + ConvBlock(32+32 → 16)
└── Conv2d(16 → 1)  ← final logits
    │
    ▼
Output: (B, 1, 256, 256)  ← logits (sigmoid at inference)
```

**Parameters:** ~6.8M (5.5M encoder + 1.3M decoder)

### 2.2 UP³RE-Net Full Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Bagging Ensemble (3 models)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Model A  │  │ Model B  │  │ Model C  │                 │
│  │ (bootstrap│  │ (bootstrap│  │ (bootstrap│                 │
│  │  sample 1)│  │  sample 2)│  │  sample 3)│                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────┐                │
│  │  Stage 1.5: Uncertainty Pre-computation │                │
│  │  σ²_ij = var(pred_A, pred_B, pred_C)_ij │                │
│  │  Saved per-batch: tmp/uncertainty/*.pt  │                │
│  └──────────────────┬──────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────────────────────────────┐                │
│  │  Stage 2: Boosting with UWACL           │                │
│  │  Model D trained with loss weighted by  │                │
│  │  w_ij = 1 + β·(1 - exp(-σ²_ij / τ_e))  │                │
│  └──────────────────┬──────────────────────┘                │
│                     │                                        │
│                     ▼                                        │
│  ┌─────────────────────────────────────────┐                │
│  │  Inference: All 4 models + confidence   │                │
│  │  mean_pred = mean(A,B,C,D)              │                │
│  │  confidence = 1 - var(A,B,C,D) / max    │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Novel Components

### 3.1 UWACL — Uncertainty-Weighted Adaptive Compound Loss

**File:** `src/losses.py:35-71`

```
L_total = (1/N) * Σ_ij [ w_ij * (λ_bce * BCE(p_ij, y_ij) + λ_dice * Dice(p_ij, y_ij)) ]
```

Where:
- `w_ij = 1 + β * (1 - exp(-σ²_ij / τ_e))` — per-pixel weight from ensemble variance
- `β = 5.0` — max weight amplification (weight ranges from 1.0 to ~5.88)
- `τ_e = 0.1 * max(0.01, 1 - 0.9 * e / total_epochs)` — dynamic temperature decay
- `λ_bce = 0.5, λ_dice = 0.5` — component mixing
- `pos_weight = 10.0` — positive class weight for BCE (addresses 937:1 imbalance)

**Patent Claim 2** covers the weight formula; **Claim 3** covers the compound loss; **Claim 4** covers the τ schedule.

### 3.2 Uncertainty Pre-computation

**File:** `src/trainer.py:110-131`

The 3 ensemble members are run sequentially on the training set. Each batch generates:
- `mean_pred`: `torch.Size([B, 1, 256, 256])` — mean prediction across 3 models
- `variance`: `torch.Size([B, 1, 256, 256])` — per-pixel variance across 3 models

Saved as `.pt` files (~10-20 MB per batch, ~10-20 GB total for full dataset).

### 3.3 Bagging Ensemble

**File:** `notebooks/04_research_training.ipynb` Cell 2

Each member is trained on a bootstrap sample (random sample with replacement) of the training volume IDs. Stratified by tumor presence: if a bootstrap sample has zero tumor-positive volumes, one volume is replaced with a tumor-positive one.

### 3.4 Uncertainty-Guided Boosting

**File:** `src/trainer.py:134-197`

`UncertaintyWeightedDataset` wraps the base training dataset with pre-computed uncertainty maps. `UncertaintyGuidedTrainer` extends `Trainer` with:
- `UncertaintyWeightedLoss` instead of `CombinedLoss`
- Dynamic τ decay: `τ_e = 0.1 * max(0.01, 1 - 0.9 * progress)`
- NaN guard: skip batches where loss exceeds 3.5 GB VRAM

---

## 4. Training Configuration

**File:** `src/config.py:146-166`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_size` | (256, 256) | Resized 2D slice dimensions |
| `batch_size` | 8 | Fits RTX 3050 Ti 4GB with mixed precision |
| `num_workers` | 4 | Parallel data loading workers |
| `lr` | 1e-3 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `epochs_stage1` | 25 | Bagging members training epochs |
| `epochs_stage2` | 25 | UWACL boosting training epochs |
| `patience` | 10 | Early stopping patience (val Dice) |
| `pos_weight` | 10.0 | BCE positive class weight |
| `dice_weight` | 0.5 | Compound loss: Dice coefficient |
| `bce_weight` | 0.5 | Compound loss: BCE weight |
| `mixed_precision` | True | `torch.cuda.amp.GradScaler` |
| `num_bagging` | 3 | Number of ensemble members |
| `uwacl_beta` | 5.0 | UWACL max weight scaling |
| `uwacl_tau` | 0.1 | UWACL initial temperature |
| `uncertainty_schedule` | linear_decay | τ decay schedule |
| `uncertainty_dir` | tmp/uncertainty | Per-batch uncertainty `.pt` files |
| `models_dir` | models/upre | Saved model checkpoints |
| `outputs_dir` | outputs/research | JSON results + figures |

**Scheduler:** CosineAnnealingLR, `T_max=epochs` (dynamically set per fit call)

---

## 5. Dataset & Data Loading

**File:** `src/data_loader.py`

### 5.1 Dataset Structure

```
Dataset/
├── Liver Img Dataset/
│   ├── Volume-000-000.png   ... Volume-000-0XX.png  (Volume 0 slices)
│   ├── Volume-001-000.png   ... Volume-001-0XX.png  (Volume 1 slices)
│   └── ... up to Volume-130-XXX.png
└── LiTS_masks/
    ├── mask-000-000.png     ... mask-000-0XX.png    (Volume 0 masks)
    ├── mask-001-000.png     ... mask-001-0XX.png    (Volume 1 masks)
    └── ... up to mask-130-XXX.png
```

**Naming convention:** `{Volume|mask}-{vol:03d}-{slice:03d}.png`

### 5.2 Data Pipeline

1. **`DataPathManager.build_index()`** — scans directories, groups slices by volume ID into `Dict[int, List[Path]]`
2. **`VolumeWiseSplitter.load_splits()`** — loads train/val/test splits from comma-separated text files (80/10/10 volume-wise, preventing slice leakage)
3. **`LiverTumor2DDataset`** — flat list of (volume_id, slice_id) tuples, loads PNG → `{'image': Tensor[1,256,256], 'mask': Tensor[1,256,256], 'volume_id': int, 'slice_id': int}`
4. **`create_2d_dataloaders()`** — wraps into `DataLoader` with configurable batch_size/num_workers/pin_memory

### 5.3 Preprocessing

**File:** `src/preprocessing.py`

- `CLAHEProcessor(clip=2.0, grid=(8,8))` — contrast-limited adaptive histogram equalization
- `AugmentedPreprocessingTransform(256, -100, 400, clahe)` — HU windowing [-100, 400] + resize 256 + CLAHE + augmentations
- `PreprocessingTransform(256, -100, 400)` — HU windowing + resize only (for val/test)

---

## 6. VRAM Management (RTX 3050 Ti 4GB)

| Phase | Max VRAM | Strategy |
|-------|----------|----------|
| Stage 1 training | ~3.2 GB | Mixed precision (`GradScaler`), batch_size=8 |
| Uncertainty pre-compute | ~0.7 GB | Sequential loading: load model → predict → unload → next |
| Stage 2 training | ~3.3 GB | Same as Stage 1 + uncertainty maps (~0.03 GB overhead) |
| Ensemble inference | ~0.7 GB | Sequential model loading |
| NaN guard | >3.5 GB | Skip batch + `torch.cuda.empty_cache()` |

---

## 7. Evaluation Metrics

**File:** `src/metrics.py`

| Metric | Formula | Description |
|--------|---------|-------------|
| Dice | `2*|P∩T| / (|P|+|T|)` | Overlap between prediction and ground truth |
| IoU | `|P∩T| / |P∪T|` | Jaccard index |
| ECE | `Σ (bin_weight * |acc - conf|)` | Expected Calibration Error (10 bins) |
| Uncertainty | `var(P_A, P_B, P_C, ...)` | Per-pixel ensemble variance (epistemic uncertainty) |

`evaluate_model()` in `src/trainer.py:200-229` handles all three evaluation modes:
- **Single model:** forward pass → sigmoid → binary metrics
- **Ensemble:** `EnsembleWrapper.predict_with_uncertainty()` → mean + variance + metrics
- **UP³RE Full:** 4 models (3 bagging + 1 boosted) → weighted ensemble

---

## 8. Notebook Structure (8 Cells)

| Cell | Name | Purpose | Key Outputs |
|------|------|---------|-------------|
| 0 | Setup | Imports, device, seed, config | `device`, `cfg` |
| 1 | Data Preparation | Load index, splits, 3 bootstraps | `train_loaders[3]`, `val_loader`, `test_loader`, `tumor_vols` |
| 2 | Stage 1: Bagging | Train 3 ensemble members | `ensemble_members[3]`, `stage1_results[]` |
| 3 | Uncertainty Pre-compute | Per-pixel variance on training set | `tmp/uncertainty/*.pt` |
| 4 | Stage 2: UWACL Boosting | Train 4th model with weighted loss | `best_boosted`, `trainer_s2` |
| 5 | Evaluation | Single vs Bagging vs UP³RE Full | `comparison_results.json` |
| 6 | Ablation | UWACL vs CombinedLoss | `ablation_results.json` |
| 7 | Figures | Bar chart, uncertainty maps, calibration | 3 PNGs in `outputs/research/figures/` |
| 8 | Export + Patent | Save summary + print patent draft | `research_summary.json`, patent markdown |

---

## 9. Output Files

| Output | Location | Format |
|--------|----------|--------|
| Model checkpoints | `models/upre/member_{0,1,2}.pth` | PyTorch state_dict |
| Boosted model | `models/upre/boosted.pth` | PyTorch state_dict |
| Uncertainty maps | `tmp/uncertainty/batch_{00000..N}.pt` | PyTorch tensors |
| Comparison results | `outputs/research/comparison_results.json` | JSON |
| Ablation results | `outputs/research/ablation_results.json` | JSON |
| Research summary | `outputs/research/research_summary.json` | JSON |
| Figures | `outputs/research/figures/` | PNG (150 DPI) |

---

## 10. Patent Claims (from `docs/RESEARCH_NOVELTY.md`)

1. **Method:** Bagging ensemble → uncertainty maps → weighted loss → boosting → combined inference
2. **Weight formula:** `w = 1 + β·(1 - exp(-σ²/τ))`
3. **UWACL loss:** Weighted compound loss (BCE + Dice)
4. **τ schedule:** Linear decay from 0.1 → 0.001 over epochs
5. **Consistency regularization** (optional future claim)
6. **System:** Memory + processor + output interface for dual output (segmentation + confidence)

---

## 11. Bug Fix History (Development Log)

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| `total_mem` | AttributeError in Cell 0 | Typo in CUDA attribute name | `total_mem` → `total_memory` |
| Dict iteration | TypeError in Cell 1 | Iterating dict keys instead of using `.get()` | `volume_index['mask_paths'].get(vid, [])` |
| `set_seed` undefined | NameError in Cell 0 | Not exported in `__init__.py` | Added `from src.utils import set_seed` |
| Emoji Unicode | UnicodeEncodeError on Windows | 🚀📋📦📊 etc. in print/log | Replaced all 15 emojis across 6 files |
| Indentation | IndentationError in Cell 1 | Replaced line had 8 spaces (rest use 4) | `        vol_masks` → `    vol_masks` |
| PIL import | NameError in Cell 1 | `Image.open()` used before `from PIL import Image` | Moved import before usage |
| Stale cache | Silent stale import | `src/__pycache__/` had old `.pyc` files | Deleted cache directory |
| Scheduler T_max | Wrong LR schedule across stages | T_max hardcoded to `epochs_stage1` in `__init__` | Dynamic `T_max=epochs` per `fit()` call |

---

## 12. File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/__init__.py` | 99 | Package API — all exports for `from src import *` |
| `src/config.py` | 166 | Paths, constants, `PHASE4_RESEARCH_CONFIG` |
| `src/data_loader.py` | 340 | `DatasetConfig`, `DataPathManager`, `VolumeWiseSplitter`, `LiverTumor2DDataset`, `create_2d_dataloaders` |
| `src/gpu_utils.py` | 113 | `setup_device`, `DEVICE`, tensor converters, GPU memory |
| `src/utils.py` | 183 | `set_seed`, `setup_logging`, `logger`, file/parallel helpers |
| `src/preprocessing.py` | 497 | HU windowing, CLAHE, `PreprocessingTransform`, `AugmentedPreprocessingTransform`, patch extraction |
| `src/models.py` | 158 | `ConvBlock`, `UNetDecoder`, `MobileNetV2UNet`, `EnsembleWrapper`, `create_model` |
| `src/trainer.py` | 232 | `Trainer`, `UncertaintyPrecomputer`, `UncertaintyGuidedTrainer`, `evaluate_model` |
| `src/losses.py` | 71 | `DiceLoss`, `CombinedLoss`, `UncertaintyWeightedLoss` |
| `src/metrics.py` | 47 | `dice_coefficient`, `iou_score`, `ensemble_uncertainty`, `calibration_error` |
| `src/visualization.py` | 403 | Plotting, report generation (legacy 3D) |
| `src/dataset.py` | 288 | Legacy `LiTSDataset` (3D NIfTI) |
| `notebooks/04_research_training.ipynb` | ~645 lines JSON | 8-cell research pipeline |
| `docs/RESEARCH_NOVELTY.md` | 296 | Prior art analysis + patent claims |
| `docs/PROJECT_OVERVIEW.md` | — | Full project documentation |

---

## 13. Running the Notebook

```bash
# 1. Clean stale cache (if any)
Remove-Item -Recurse -Force src/__pycache__ -ErrorAction SilentlyContinue

# 2. Launch Jupyter
jupyter notebook notebooks/04_research_training.ipynb

# 3. In browser: Kernel → Restart & Run All
```

**Expected runtime per cell (RTX 3050 Ti):**
- Cell 0-1: ~30s (data loading)
- Cell 2: ~1-2 hrs (3 members × 25 epochs)
- Cell 3: ~15-30 min (uncertainty pre-compute)
- Cell 4: ~30-45 min (boosting 25 epochs)
- Cell 5-8: ~5 min (evaluation + export)

---

*Generated from project source files. Last updated: May 2026.*

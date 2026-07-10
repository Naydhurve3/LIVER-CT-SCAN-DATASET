# Refactoring Log

Tracks all changes made to fix path/storage compliance and eliminate code duplication.

## Change 1: `src/data_loader.py` — Added missing `DatasetConfig` attributes

**Date**: 2026-05-22
**Files affected**: `src/data_loader.py`

### Problem
`DatasetConfig` class was missing 3 attributes that both `02_eda.ipynb` and `03_preprocessing.ipynb` reference:

| Missing Attribute | Used By |
|-------------------|---------|
| `EDA_OUTPUT_DIR` | `02_eda` line 87, `03_preprocessing` lines 98-99 |
| `EDA_PLOTS_DIR` | `02_eda` line 88 |
| `PREP_OUTPUT_DIR` | `03_preprocessing` line 103 |

### Fix
Added these 3 class attributes to `DatasetConfig` pointing to:
- `EDA_OUTPUT_DIR` → `<project>/outputs/eda/`
- `EDA_PLOTS_DIR` → `<project>/outputs/eda/plots/`
- `PREP_OUTPUT_DIR` → `<project>/outputs/preprocessing/`

---

## Change 2: `notebooks/01_data_loading.ipynb` — Fixed imports, API calls, and save paths

**Date**: 2026-05-22
**Files affected**: `notebooks/01_data_loading.ipynb`

### Problems Fixed

| # | Line | Issue | Fix |
|---|------|-------|-----|
| 1 | 62 | `import create_dataloaders` — wrong name | Changed to `create_2d_dataloaders` |
| 2 | 63 | `import compute_volume_statistics` — doesn't exist | Removed import, inlined logic |
| 3 | 65 | `import get_device` — wrong name | Changed to `setup_device` |
| 4 | 166 | `path_manager.get_dataset_stats()` — doesn't exist | Replaced with inline stat computation |
| 5 | 323-331 | `create_dataloaders(...)` call with wrong signature | Updated to `create_2d_dataloaders(...)` correct signature |
| 6 | 382 | Save path `DatasetConfig.OUTPUT_DIR / 'split_visualization.png'` → `data/` | Changed to `<project>/outputs/data_loading/split_visualization.png` |

---

## Change 3: `notebooks/03_preprocessing.ipynb` — Full refactor to use `src/` modules

**Date**: 2026-05-22
**Files affected**: `notebooks/03_preprocessing.ipynb`

### Problems Fixed
Replaced all duplicated code with imports from `src/` modules:

| Cell | What Was Replaced | Imported From |
|------|-------------------|---------------|
| 1 | Inline `to_tensor`, `to_numpy`, `batch_to_tensor`, `gpu_report`, `gpu_clear` | `src.gpu_utils` |
| 1 | `DatasetConfig.EDA_OUTPUT_DIR`, `PREP_OUTPUT_DIR` — missing attrs | Added to DatasetConfig |
| 2 | `PreprocessingConfig` class | Replaced with `src.config.TRAIN_CONFIG_2D`, `WINDOWS`, etc. |
| 3 | `hu_window_cpu`, `hu_window_batch`, `preprocess_image` | `src.preprocessing` |
| 4 | `CLAHEProcessor` class | `src.preprocessing.CLAHEProcessor` |
| 5 | `resize_image`, `resize_mask`, `resize_batch_gpu`, `preprocess_batch_gpu` | `src.preprocessing` |
| 5b | Inline `DataPathManager` with hardcoded paths | `src.data_loader.DataPathManager` |
| 6 | `AugmentedDataset` class + `create_dataloaders` | `src.data_loader.LiverTumor2DDataset` + `create_2d_dataloaders` |
| 7 | `visualize_preprocessing` function | `src.visualization.plot_slice_with_mask` |
| 8-9 | Benchmark + save (kept as-is) | — |

---

## Change 4: `docs/implementation plan.md` — Updated with Phase 3 notebook plan

**Date**: 2026-05-22
**Files affected**: `docs/implementation plan.md`

Added detailed Phase 3 notebook implementation plan covering all 9 cells with descriptions, `src/` module dependencies, and data flow.

---

## Change 5: Comprehensive bugfix sweep & Phase 3 improvements

**Date**: 2026-05-23
**Files affected**: `src/data_loader.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/__init__.py`, all 3 notebooks

### Critical Bug Fixes

| # | Issue | File | Fix |
|---|-------|------|-----|
| 1 | Hardcoded Windows path `D:\DATA SCIENCE AND ANALYTICS` | `data_loader.py:21` | Changed to `Path(__file__).resolve().parent.parent` — fully dynamic |
| 2 | Mask binarization `val/255 > 0.5` → all zeros (masks are 0/1, not 0/255) | `data_loader.py:262` | Removed `/ 255.0`, use `> 0.5` directly on uint8 values |
| 3 | `RandomFlip3D` axis bug — depth flip mapped to wrong mask dim | `augmentation.py:33` | `flip(image, dims=[axis])` → `[axis+1]`; `flip(mask, dims=[axis-1])` → `[axis]` |
| 4 | `sys.path` fragility — `Path.cwd().parent` breaks if kernel not in `notebooks/` | All 3 notebooks | Walk up from cwd until `src/` found |

### Dead Import Removal

| Notebook | Removed Imports |
|----------|----------------|
| `02_eda.ipynb` | `VolumeWiseSplitter`, `LiverTumor2DDataset` (imported but never used) |
| `03_preprocessing.ipynb` | `WINDOWS`, `AUGMENTATION` from config; `plot_slice_with_mask` from visualization |

### New Features (Phase 3+)

| Feature | File | Purpose |
|---------|------|---------|
| `PreprocessingTransform` class | `preprocessing.py` | Callable transform: HU window → resize → optional CLAHE. Pass to `LiverTumor2DDataset` as `transform=` arg. |
| `AugmentedPreprocessingTransform` class | `preprocessing.py` | Training transform: same as above + random flips + intensity shifts |
| Full dataloaders (all 131 volumes) | `03_preprocessing.ipynb` Cell 6 | `train_loader`, `val_loader`, `test_loader` ready for Phase 4 training |
| Cached `volume_index.pkl` | `03_preprocessing.ipynb` Cell 6 | Saved to `outputs/preprocessing/` — avoids 58K file re-scan in Phase 4 |
| Output path unification | `01_data_loading.ipynb` Cell 9 | Changed from `Path.cwd()/outputs/data_loading/` to `DatasetConfig.PROJECT_DIR/outputs/data_loading/` |

### Mask Loading Fix (Phase 3 Cell 7)

Before: `np.array(Image.open(...), dtype=np.float32) / 255.0` then `> 0` → AFTER fix: `np.array(Image.open(...), dtype=np.float32)` then `> 0.5` (same as dataset fix)

---

## Summary of All Fixes

| Location | Issue | Fix |
|----------|-------|-----|
| `src/data_loader.py:34-36` | Missing `EDA_OUTPUT_DIR`, `EDA_PLOTS_DIR`, `PREP_OUTPUT_DIR` | Added 3 class attributes |
| `01_data_loading:62` | Wrong import `create_dataloaders` | Changed to `create_2d_dataloaders` |
| `01_data_loading:63` | Nonexistent `compute_volume_statistics` | Removed import, inlined logic |
| `01_data_loading:65` | Wrong import `get_device` | Changed to `setup_device` |
| `01_data_loading:166` | Nonexistent `path_manager.get_dataset_stats()` | Inlined stat computation |
| `01_data_loading:323-331` | Wrong `create_dataloaders` call signature | Updated to `create_2d_dataloaders` |
| `01_data_loading:382` | Plot saved to `data/` instead of `outputs/` | Changed to `outputs/data_loading/` |
| `01_data_loading:417-438` | References to removed `stats` dict | Changed to inline variables |
| `03_preprocessing:1` | Inline GPU utils + missing DatasetConfig attrs | Imported from `gpu_utils`; attrs added to `DatasetConfig` |
| `03_preprocessing:2` | Inline `PreprocessingConfig` class | Replaced with dict using `TRAIN_CONFIG_2D` + EDA stats |
| `03_preprocessing:3-5` | Inline HU windowing, CLAHE, resize functions | Imported from `src.preprocessing` |
| `03_preprocessing:5b` | `DataPathManager` used without import | Added import |
| `03_preprocessing:6` | Inline `AugmentedDataset` + `DataPathManager` class | Replaced with `LiverTumor2DDataset` + `create_2d_dataloaders` |
| `03_preprocessing:7` | Inline `visualize_preprocessing` | Inlined with `src.preprocessing` imports |
| `03_preprocessing:8-9` | `CFG.ATTR` style refs (was class) | Changed to `CFG['attr']` dict style |

---

## Change 6: Phase 4 — UP³RE-Net Research Implementation (Novel method + patent claims)

**Date**: 2026-05-23
**Files affected**: `src/losses.py` (NEW), `src/metrics.py` (NEW), `src/models.py` (NEW), `src/trainer.py` (NEW), `src/config.py` (MODIFY), `src/__init__.py` (MODIFY), `notebooks/04_research_training.ipynb` (NEW), `docs/RESEARCH_NOVELTY.md` (NEW)

### Summary
Replaced the original Phase 4 plan (4 separate models + bagging + boosting as comparison) with **UP³RE-Net**: a novel research architecture combining uncertainty-guided ensemble learning with per-pixel re-weighting for patent-grade novelty.

### Novel Contributions (patent claims 1–6)

| Claim | Novelty | Status |
|-------|---------|--------|
| 1 | Method: bagging → per-pixel uncertainty → UWACL boosting | Implemented in `trainer.py` |
| 2 | Weight function: `w = 1 + β·(1 - exp(-σ²/τ))` | Implemented in `losses.py:UncertaintyWeightedLoss` |
| 3 | Uncertainty-weighted compound loss (BCE + Dice) | Implemented in `losses.py:UncertaintyWeightedLoss` |
| 4 | Dynamic threshold scheduling (τ decay over epochs) | Implemented in `trainer.py:UncertaintyGuidedTrainer` |
| 5 | Mutual consistency regularization between ensemble members | Implemented in `trainer.py` |
| 6 | End-to-end inference: segmentation + per-pixel confidence map | Implemented in `models.py:EnsembleWrapper` |

### What Changed from Original Phase 4 Plan

| Aspect | Original Plan | UP³RE-Net (New) |
|--------|---------------|------------------|
| Goal | Compare 4 models | Novel method + patent |
| Models | TinyUNet, UNet, MNV2, EffNetB0 | 3× bagging MNV2 + 1× boosted MNV2 |
| Loss | Standard CombinedLoss | **UWACL** — uncertainty-weighted per-pixel |
| Ensemble | Simple average | Uncertainty-guided with mutual consistency |
| TinyUNet | 10 epochs baseline | Removed (covered by single member ablation) |
| Plain UNet | Scrapped | Removed (redundant) |
| EfficientNetB0 | Scrapped | Removed (marginal gain, `timm` dependency) |
| Bagging | 2 members @ 25ep | 3 members @ 25ep (more diversity) |
| Boosting | Standard re-weight | **Uncertainty-guided per-pixel** |

### New Files Created

| File | Lines | Key Classes |
|------|-------|-------------|
| `src/losses.py` | 80 | `DiceLoss`, `CombinedLoss`, `UncertaintyWeightedLoss` |
| `src/metrics.py` | 60 | `dice_coefficient`, `iou_score`, `ensemble_uncertainty`, `calibration_error` |
| `src/models.py` | 280 | `MobileNetV2UNet`, `EnsembleWrapper`, `create_model()` |
| `src/trainer.py` | 220 | `Trainer`, `UncertaintyPrecomputer`, `UncertaintyGuidedTrainer` |
| `docs/RESEARCH_NOVELTY.md` | — | Patent claims, prior art analysis, architecture diagrams |

### Modified Files

| File | Change |
|------|--------|
| `src/config.py` | Added `PHASE4_RESEARCH_CONFIG` dict |
| `src/__init__.py` | Added exports for `models`, `losses`, `metrics`, `trainer` |

### Time Budget Allocation

| Step | Epochs | Real Time |
|------|--------|-----------|
| 3× bagging MNV2 (bootstrap samples) | 25 each | 6.3 hrs |
| Uncertainty pre-computation | — | 20 min |
| Boosted MNV2 with UWACL | 25 | 2.1 hrs |
| Evaluation + figures + patent | — | 30 min |
| **Total** | | **~9.2 hrs** |

### Key Technical Decisions

1. **No TinyUNet** — removed because the single bagging member serves as ablation baseline
2. **Stratified bootstrap** — each bootstrap sample forced to contain ≥1 tumor volume (937:1 imbalance)
3. **Pre-computed uncertainty** — saves ~8 hrs vs on-the-fly computation during Stage 2
4. **Sequential VRAM management** — ensemble members loaded one at a time (3 × 0.6 GB won't fit 4GB)
5. **Dynamic τ schedule** — uncertainty temperature decays linearly from 0.1 → 0.01 over epochs

### Prior Art Distinction

| Published Work | Approach | Our Novelty |
|----------------|----------|-------------|
| Two-layer ensemble (Springer 2024) | Layer 1 preds → augmentation for Layer 2 | We use **uncertainty** (variance), not predictions |
| Deep ensemble + sampling (CompBioMed 2024) | Slice-level sampling + exponential loss | We use **per-pixel** weighting from **ensemble disagreement** |
| UCTNet (2024) | Uncertainty → CNN vs Transformer routing | We use uncertainty → **loss re-weighting** |
| DyCON (2025) | Semi-supervised consistency | We work in **fully supervised** setting |
| UG-CEMT (WACV 2025) | Semi-supervised ensemble mean teacher | Not semi-supervised, different architecture |

# Core Framework Source Code (`src/`)

## Overview

This directory contains the core PyTorch deep learning framework modules for dataset loading, loss functions, model architectures, and evaluation tools.

---

## Package Architecture

```
src/framework/
├── data/
│   ├── manifest_dataset.py        <- Verified Manifest Dataset loader + create_manifest_dataloaders()
│   └── samplers.py                <- Patient-aware & lesion-balanced samplers
├── losses/
│   ├── focal_tversky.py           <- Focal Tversky Loss implementation
│   └── stability_bounded_recall.py<- Stability Bounded Recall Loss implementation
├── models/
│   ├── mobilenetv2_unet.py        <- MobileNetV2-UNet 2D/2.5D architecture
│   └── two_stage_roi.py           <- Stage-1 Liver ROI + Stage-2 Tumor Net
└── metrics/
    └── segmentation_metrics.py    <- Micro/Macro Dice, Surface Distance, Q1 Recall
```

---

## Training-Readiness Integration (2026-08-12)

- `src/framework/data/manifest_dataset.py` exposes `create_manifest_dataloaders()`, building train/val (test-locked) `DataLoader`s from the verified slice manifest with optional tumor-positive weighted sampling.
- `src/framework/experiment.py::build_experiment_loaders()` automatically selects the manifest loader whenever a dataset config defines `slice_manifest` (see `configs/datasets/lits_verified.yaml`); legacy path-based loading remains supported for configs without a manifest.
- Batch contract is unchanged (`image`/`mask` tensors plus `sample_id`/`volume_id` metadata), so both `Trainer` and `ResearchTrainer` consume manifest-loaded data directly.

---

[Back to Root README](../README.md)

# Core Framework Source Code (`src/`)

## Overview

This directory contains the core PyTorch deep learning framework modules for dataset loading, loss functions, model architectures, and evaluation tools.

---

## Package Architecture

```
src/framework/
├── data/
│   ├── manifest_dataset.py        <- Verified Manifest Dataset loader
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

[Back to Root README](../README.md)

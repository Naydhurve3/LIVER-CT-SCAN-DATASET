# Experiment & Dataset Configurations

## Overview

Contains YAML configuration files specifying dataset paths, split definitions, preprocessing profiles, hyperparameter choices, and model architectures.

---

## Configuration Index

- `datasets/lits_verified_eda.yaml`: Standard configuration for canonical build `build_corrected_20260713_214847_v2`.
- `datasets/lits_verified.yaml`: **Training-ready** dataset config for the same canonical build (`training_ready: true`), consumed by manifest-driven experiment configs.
- `experiments/research_manifest_baseline.yaml`: Manifest-driven research baseline (MobileNetV2-UNet + FocalDice) that loads training/validation data exclusively via `VerifiedManifestDataset` from the verified build.
- `splits/`: CSV and TXT files defining volume assignment across Train (104), Validation (13), and Sealed Test (14) sets.

---

## Change Log

### 2026-08-12 — Manifest-driven training wiring
- Added `configs/datasets/lits_verified.yaml` (verified training-ready dataset: manifest hash `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`).
- Added `configs/experiments/research_manifest_baseline.yaml`, which inherits `lits_verified` and routes loading through `VerifiedManifestDataset`.
- `build_experiment_loaders()` (in `src/framework/experiment.py`) now auto-selects the manifest loader whenever a dataset config defines `slice_manifest`; legacy path-based loading remains available when no manifest is configured.
- `tools/train.py` passes `limit_slices` for dry runs and falls back to `last_checkpoint.pth` when a tiny dry run never records a valid best checkpoint.

---

[Back to Root README](../README.md)

# MedSegX Research Changelog

> Canonical record of all architectural changes, enhancements, bug fixes, and experimental results.
> Every entry includes date, file paths, and rationale for reproducibility.

---

## Table of Contents

1. [Sprint 1: Infrastructure](#sprint-1-infrastructure)
2. [Sprint 2: Migration](#sprint-2-migration)
3. [Sprint 3: Evaluation](#sprint-3-evaluation)
4. [Sprint 4: Baseline Validation](#sprint-4-baseline-validation)
5. [Sprint 5: Research Preparation](#sprint-5-research-preparation)
6. [Current Sprint](#current-sprint)

---

## Sprint 1: Infrastructure

**Date:** 2026-07-06 — 2026-07-06

### Changes Made

| # | Change | Files | Rationale |
|---|--------|-------|-----------|
| 1.1 | Created project skeleton with 25 Python packages | All `__init__.py` files | Module structure for framework/research separation |
| 1.2 | Implemented core module with zero external deps | `src/framework/core/{constants,exceptions,interfaces,registry,config,reproducibility,factory}.py` | Foundation layer that all other modules depend on |
| 1.3 | Created Registry pattern | `src/framework/core/registry.py` | Config-driven dispatch without if/elif chains |
| 1.4 | Created layered config system | `src/framework/core/config.py` | YAML inheritance: defaults → dataset → model → experiment → CLI |
| 1.5 | Created 16 config YAMLs | `configs/` | Defaults for training, optimizer, augmentation, preprocessing, evaluation |
| 1.6 | Created 4 ADRs | `docs/architecture/ADR/` | Documenting ADR-001 (Core), ADR-002 (Config), ADR-003 (Reproducibility), ADR-007 (Framework/Research separation) |
| 1.7 | Environment configuration | `.gitignore`, `.env.example`, `requirements.txt`, `environment.yaml` | Project setup basics |

### Key Decisions

- Architecture v1.0 frozen — no structural changes without a new ADR
- `core/` module has zero dependencies on other MedSegX modules
- All models/losses through registries via decorator, never if/elif chains

---

## Sprint 2: Migration

**Date:** 2026-07-06 — 2026-07-06

### Changes Made

| # | Change | Files | Rationale |
|---|--------|-------|-----------|
| 2.1 | Ported 33 Python source files from old Liver project | `src/framework/{models,losses,data,training,evaluation,analytics,uncertainty,utils}/` | Migrate all existing functionality into new structure |
| 2.2 | Added registry decorators to all models and losses | All model/loss files | Enable config-driven dispatch |
| 2.3 | Copied 3 trained checkpoints | `models/upre/member_{0,1,2}.pth` | Preserve old project results |
| 2.4 | Archived old project | `Archive/Liver_Project_v1/` | Clean separation from new codebase |
| 2.5 | Migrated data splits | `data/splits/` | 104/13/14 train/val/test split |

### Key Decisions

- Old project archived, not deleted — reference for comparison
- All migrated code uses registry pattern (no direct imports)

---

## Sprint 3: Evaluation

**Date:** 2026-07-06 — 2026-07-06

### Changes Made

| # | Change | Files | Rationale |
|---|--------|-------|-----------|
| 3.1 | Surface distance metrics | `src/framework/evaluation/surface_metrics.py` | HD95, ASD, NSD with scipy KDTree |
| 3.2 | Bootstrap statistics + Wilcoxon | `src/framework/evaluation/statistics.py` | 1000 resamples for CI, Wilcoxon signed-rank test |
| 3.3 | Cross-validation runner | `src/framework/evaluation/cross_validation.py` | K-fold volume-wise split + CrossValidator |
| 3.4 | Metrics tracker + MLflow | `src/framework/evaluation/tracker.py` | Local JSON logging + MLflow experiment tracking with calibration plots |

### Verification

- 24 new tests added, 120 total tests passing

---

## Sprint 4: Baseline Validation

**Date:** 2026-07-06 — 2026-07-06

### Changes Made

| # | Change | Files | Rationale |
|---|--------|-------|-----------|
| 4.1 | CUDA torch installation | Environment: torch 2.5.1+cu124 | GPU acceleration on RTX 3050 Ti |
| 4.2 | Validation script | `tools/validate.py` | Load old checkpoints, compute metrics, benchmark |
| 4.3 | NumPy version pin | `<2` in requirements | Compatibility with existing code |

### Key Results

| Metric | Value |
|--------|-------|
| Dice (3-member ensemble) | 0.8508 ± 0.0483 (4165 val slices) |
| Member 0 | 0.859 |
| Member 1 | 0.906 |
| Member 2 | 0.788 |
| Inference time | 5-10 ms/slice |
| VRAM usage | 415 MB (batch_size=1) |
| Parameters | 6.8M (per model) |

### Environment Setup

```yaml
Python: 3.11.15 (via uv)
CUDA: 13.1 (driver 592.00)
torch: 2.5.1+cu124
GPU: RTX 3050 Ti (4 GB VRAM)
OS: Windows 11
```

---

## Sprint 5: Research Preparation

**Date:** 2026-07-06 — 2026-07-06

### Changes Made

| # | Change | Files | Rationale |
|---|--------|-------|-----------|
| 5.1 | Research directory structure | `research/{faupnet,uwacl,ablation}/` | Paper-specific code and design docs |
| 5.2 | FAUP-Net design doc | `research/faupnet/DESIGN.md` | Scope: gated skip connections on last 2 encoder blocks only |
| 5.3 | UWACL-v2 design doc | `research/uwacl/DESIGN.md` | Multi-scale uncertainty aggregation + edge-awareness |
| 5.4 | FAUPNet model implementation | `src/research/faupnet/model.py` | GatedSkipConnection, UncertaintyHead (Conv3×3→1 + dropout p=0.1, <50K params) |
| 5.5 | UWACLv2MultiScale loss | `src/research/uwacl/loss_v2.py` | Multi-scale uncertainty (1×1/3×3/7×7) + Laplacian edge loss + tau decay |
| 5.6 | Research module init files | `src/research/__init__.py`, `{faupnet,uwacl}/__init__.py` | Proper exports for factory auto-import |
| 5.7 | Ablation runner | `tools/ablate.py` | 10 experiments, zero_shot + train modes, JSON results |
| 5.8 | Ablation config YAMLs | `configs/experiments/{faupnet_scoped,ablation_baseline,ablation_faupnet}.yaml`, `configs/models/faupnet.yaml` | Config templates for all experiments |

### Bugs Fixed

| # | Bug | File | Fix | Impact |
|---|-----|------|-----|--------|
| 5.9 | `_build_from_registry` mutated config dict via `pop("name")`, causing KeyError on subsequent calls | `src/framework/core/factory.py` | Changed `config.pop("name")` to `cfg = dict(config); cfg.pop("name")` | All ablation experiments were failing with `ERROR: 'name'` |
| 5.10 | FAUPNet encoder channel mismatch: stem output 32ch but first encoder stage expected 16ch | `src/research/faupnet/model.py` | Aligned encoder stages with `MobileNetV2UNet`'s block splitting: `features[0]`, `[1:3]`, `[3:5]`, `[5:8]`, `[8:15]`, `[15:]` | All 5 FAUPNet forward tests were failing |
| 5.11 | First decoder block channel count wrong: got 1280 but needed 1440 (1280 + 160 skip) | `src/research/faupnet/model.py` | Changed `in_ch = rev_enc[i] if i == 0 else dec_channels[i-1]` then `in_ch += skip_ch` | Fixes FAUPNet decoder dimension mismatch |
| 5.12 | DataLoader hang after ~4 iterations on Windows (num_workers=4 + persistent_workers=True deadlock) | `tools/ablate.py`, `src/framework/data/lits_dataset.py` | Set `num_workers=0`; recreate DataLoaders per experiment | Ablation suite now runs all 10 experiments without hanging |
| 5.13 | Train-mode ablation used `transform_val=transform` for training DataLoader, feeding 512×512 images | `tools/ablate.py` | Changed to `transform_train=transform, transform_val=transform` | Training data now properly resized to 256×256 |
| 5.14 | GatedSkipConnection API mismatch: took 2 args (in_ch, out_ch) but only needed 1 (in_ch) | `src/research/faupnet/model.py` | Removed `out_channels` parameter, simplified forward to single-arg | Test `test_gated_skip` was failing with TypeError |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| FAUP-Net scoped to gated skip connections on last 2 encoder blocks only | VRAM optimization per RTX 3050 Ti 4GB limit; avoids full multi-level complexity |
| Uncertainty head: single conv + dropout, not Monte Carlo | Single forward pass for inference speed; no sampling overhead |
| Multi-scale uncertainty: 3 scales (1×1, 3×3, 7×7) with learned weights [0.4, 0.35, 0.25] | Captures uncertainty at different receptive fields |
| Edge loss coefficient λ=0.1 | Small enough to avoid dominating training, large enough to affect boundary predictions |
| Ablation: 10 experiments covering model × loss × ensemble | Directly maps to paper claims: gate effect, loss effect, ensemble effect |
| `uv` for Python management instead of conda | Faster, no conda env needed; `.venv\Scripts\python.exe` for all commands |

### Ablation Results (1-epoch train, 2 train volumes, 1 val volume volume-104 — 781 slices)

| Experiment | Model | Loss | Ensemble | Dice |
|-----------|-------|------|----------|------|
| baseline | MobileNetV2UNet | combined | 1 | 0.844 |
| baseline_uwaclv1 | MobileNetV2UNet | uwacl_v1 | 1 | 0.844 |
| baseline_uwaclv2 | MobileNetV2UNet | uwacl_v2 | 1 | 0.530 |
| ensemble3 | MobileNetV2UNet | combined | 3 | 0.844 |
| ensemble3_uwaclv1 | MobileNetV2UNet | uwacl_v1 | 3 | 0.844 |
| ensemble3_uwaclv2 | MobileNetV2UNet | uwacl_v2 | 3 | 0.530 |
| faupnet | FAUPNet | combined | 1 | 0.844 |
| faupnet_uwaclv1 | FAUPNet | uwacl_v1 | 1 | 0.844 |
| faupnet_uwaclv2 | FAUPNet | uwacl_v2 | 1 | 0.442 |
| faupnet_ensemble3_uwaclv2 | FAUPNet | uwacl_v2 | 3 | 0.442 |

**Key observations:**
- Dice=0.844 is the "all-background" baseline (model outputs all <0.5, matching background-dominant data)
- UWACLv2 (with edge loss) pushes models out of all-background equilibrium even at epoch 1 — edge loss creates gradients that activate foreground prediction
- FAUPNet + UWACLv2 achieves lowest Dice (0.442) — gating mechanism combined with edge loss creates conflicting gradients in early training
- Longer training needed for meaningful comparison (Sprint 6)

### Test Results

- **Total tests:** 139 (120 core + 19 research)
- **FAUPNet tests (7):** all passing — forward, predict, uncertainty maps, uncertainty head, gated skip, registration, different gate levels
- **UWACLv2 tests (8):** all passing — forward, with uncertainty, tau schedule, multi-scale, edge loss, set_tau, registration, edge-only mode
- **Ablation tests (4):** all passing — registry, config build, experiment enumeration, zero_shot run

---

## Current Sprint

_To be filled as Sprint 6 progresses._

---

## Appendix: Environment

### Python Environment (created via `uv`)

```
Python: 3.11.15
uv: C:\Users\alanm\.local\bin\uv.exe
Venv: MedSegX/.venv/
```

### Key Packages

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.5.1+cu124 | Deep learning framework (CUDA 12.4) |
| numpy | 1.26.4 | Numerical computing (pinned <2) |
| torchvision | 0.20.1 | Pretrained backbones (MobileNetV2) |
| scipy | 1.14.1 | KDTree for surface distance metrics |
| opencv-python | 4.10.0 | Image preprocessing / CLAHE |
| mlflow | 2.19.0 | Experiment tracking |
| scikit-learn | 1.6.1 | Bootstrapping utilities |
| pillow | 11.1.0 | PNG image I/O |

### CUDA Configuration

```
CUDA Driver: 592.00
CUDA Runtime: 13.1
GPU: NVIDIA GeForce RTX 3050 Ti (4 GB VRAM)
Compute Capability: 8.6
Available: True
```

### Dataset

```
Source: Medical Segmentation Decathlon — Liver (LiTS)
Total volumes: 131
Total slices: 58,638 (PNG format)
Split: 104 train / 13 val / 14 test
Image size: 512×512 (resized to 256×256)
Class distribution: background ~93.7%, liver ~5.9%, tumor ~0.4%
Location: D:\DATA SCIENCE AND ANALYTICS\Dataset\
```

---

*Last updated: 2026-07-06*

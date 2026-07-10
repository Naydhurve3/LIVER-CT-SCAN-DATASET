# Liver Tumor Segmentation — Project Overview

> Dataset facts in `Practice/` and `docs/DATA_REFERENCE.md` take precedence over
> older values in this historical overview.

**Hardware:** ASUS TUF F15 · RTX 3050 Ti 4GB · 16GB RAM  
**Framework:** PyTorch 2.x  
**Task:** Binary segmentation (liver+tumor vs background) from 2D CT slices

---

## Dataset

| Item | Path (dynamic) |
|------|----------------|
| Images (58,638 PNGs) | `Dataset/Liver Img Dataset/Volume-{vid:03d}-{slice:03d}.png` |
| Masks (58,638 PNGs) | `Dataset/LiTS_masks/mask-{vid:03d}-{slice:03d}.png` |
| Source (Kaggle) | [LiTS PNG](https://www.kaggle.com/datasets/andrewmvd/lits-png) |

### Key Statistics (from Phase 2 EDA)

| Metric | Value |
|--------|-------|
| Total volumes | 131 (IDs 0–130) |
| Total slices | 58,638 |
| Volumes with masks | 131 (100%) |
| Image storage | 512×512 RGBA PNG; converted to grayscale |
| Mask storage | 256×256 RGB PNG; converted to one-channel binary |
| Slices with tumor | 12.14% |
| Class imbalance | ~937:1 (train), ~1381:1 (val), ~369:1 (test) |
| Mean tumor coverage | 0.087% per slice |

### Volume Split (80/10/10 — sorted, volume-wise)

| Split | Volumes | Slices | Mean slices/vol |
|-------|---------|--------|----------------|
| Train | 0–103 (104) | 40,667 | 391.0 |
| Val | 104–116 (13) | 10,685 | 821.9 |
| Test | 117–130 (14) | 7,286 | 520.4 |

**Note:** Val and test volumes are systematically larger and have more tumor pixels per volume than training. This may affect evaluation metrics.

### Outlier Volumes (28 identified)
- **Few slices (<100):** [0, 31, 45, 54, 66, 71, 72, 75, 77] — 9 volumes
- **Many slices (>800):** [4, 17, 18, 27, 83, 87, 88, 92, 94, 95, 105, 108, 110, 113, 114, 115, 116, 117, 127] — 19 volumes

---

## Pipeline Phases

### Phase 1: Data Loading (COMPLETE)
**Notebook:** `notebooks/01_data_loading.ipynb`
- Builds volume index by scanning 58K PNG files
- Creates volume-wise train/val/test split (no data leakage)
- Saves split files to `data/splits/`
- Saves summary to `data/metadata/phase1_summary.json`

### Phase 2: EDA (COMPLETE)
**Notebook:** `notebooks/02_eda.ipynb`
- Intensity distribution analysis
- Tumor coverage analysis (12.14% slices with tumor)
- Class imbalance per split (937:1 to 1381:1)
- Morphology analysis (connected components, aspect ratios)
- PNG intensity analysis; numeric HU windowing is disabled because calibrated
  Hounsfield units are not retained in this export
- CLAHE comparison
- 2.5D context analysis → recommended context=3
- Outlier identification (28 volumes)
- All plots saved to `outputs/eda/plots/` (13 PNGs)

### Phase 3: Preprocessing (COMPLETE)
**Notebook:** `notebooks/03_preprocessing.ipynb`
- Preserve already-windowed 8-bit PNG intensities
- CLAHE: clip=2.0, grid=8×8
- Resize: 512×512 → 256×256
- GPU batch preprocessing benchmark
- Creates full dataset dataloaders with transforms
- Exports preprocessing config to `outputs/preprocessing/preprocessing_config.json`
- Caches volume index to `outputs/preprocessing/volume_index.pkl`

### Phase 4: UP³RE-Net Research & Training (IMPLEMENTED)
**Notebook:** `notebooks/04_research_training.ipynb`
**Novel method:** Uncertainty-Propagated Per-Pixel Reweighting Ensemble Network

**Architecture:** 3× MobileNetV2U-Net bagging ensemble + 1× boosted MobileNetV2U-Net with uncertainty-guided reweighting

**Novel components (patent pending):**
- **UWACL**: Uncertainty-Weighted Adaptive Compound Loss — per-pixel loss weighting from ensemble variance
- **Per-pixel uncertainty feedback**: ensemble disagreement → pixel weight map → boosting
- **Dynamic threshold scheduling**: uncertainty temperature decays over epochs
- **End-to-end confidence maps**: output segmentation + per-pixel confidence

**Training budget:** ~9.2 hrs (3 bagging + 1 boosted, 25 epochs each)
- Input: (1, 256, 256), batch size: 8
- 3 stratified bootstrap samples from 104 train volumes
- Uncertainty pre-computed (20 min) then boosted training (2.1 hrs)

**Outputs:**
- `models/upre/` — 4 trained checkpoints + inference config
- `outputs/research/` — comparison results, ablation, figures
- `outputs/research/patent_draft.md` — 6 patent claims
- `outputs/research/figures/` — bar chart, uncertainty maps, calibration plot

### Phase 5: Evaluation (PLANNED)
### Phase 6: Deployment (PLANNED)

---

## Preprocessing Parameters (for Phase 4)

| Parameter | Value | Source |
|-----------|-------|--------|
| Target size | 256×256 | `TRAIN_CONFIG_2D` |
| HU window | [-100, 400] | Phase 2 EDA |
| CLAHE clip | 2.0 | Phase 3 |
| CLAHE grid | 8×8 | Phase 3 |
| Batch size | 8 | `TRAIN_CONFIG_2D` |
| Num workers | 4 | `TRAIN_CONFIG_2D` |
| 2.5D context | 3 slices (disabled) | Phase 2 EDA |

Transforms available in `src.preprocessing`:
- `PreprocessingTransform` — HU window → resize → optional CLAHE (for val/test)
- `AugmentedPreprocessingTransform` — same + random flips + intensity shifts (for train)

---

## Project Structure

```
Liver/
├── src/                            # Source modules
│   ├── __init__.py                 # Public API exports
│   ├── config.py                   # Training configs, windows, paths
│   ├── data_loader.py              # DatasetConfig, DataPathManager, Dataset, Dataloaders
│   ├── preprocessing.py            # HU windowing, CLAHE, resize, transforms
│   ├── augmentation.py             # 3D augmentation transforms (RandomFlip3D, etc.)
│   ├── gpu_utils.py                # Device setup, tensor transfer, memory tools
│   ├── losses.py                   # DiceLoss, CombinedLoss, UWACL (★ novel)
│   ├── metrics.py                  # dice, iou, ensemble_uncertainty, calibration_error
│   ├── models.py                   # MobileNetV2UNet, EnsembleWrapper, factory (★ novel)
│   ├── trainer.py                  # Trainer, UncertaintyGuidedTrainer (★ novel)
│   ├── visualization.py            # Plotting utilities (7 functions)
│   ├── utils.py                    # Logging, file I/O, seeding
│   └── dataset.py                  # (Legacy) 3D LiTS NIfTI dataset
├── notebooks/
│   ├── 01_data_loading.ipynb       # Phase 1: volume index & splits
│   ├── 02_eda.ipynb                # Phase 2: exploratory data analysis
│   ├── 03_preprocessing.ipynb      # Phase 3: preprocessing pipeline
│   └── 04_research_training.ipynb  # Phase 4: UP³RE-Net research pipeline
├── data/
│   ├── splits/                     # train/val/test_volumes.txt
│   └── metadata/                   # phase1_summary.json, phase2_statistics.json
├── outputs/
│   ├── eda/
│   │   ├── statistics.json         # Full EDA report
│   │   ├── enhanced_volume_stats.json  # HU windows, context, outliers
│   │   ├── outlier_volumes.json    # Outlier volume IDs
│   │   └── plots/                  # 13 EDA visualization PNGs
│   └── preprocessing/
│       ├── preprocessing_config.json  # Pipeline parameters
│       ├── volume_index.pkl        # ~8.5MB cached index (avoids 58K file re-scan)
│       └── plots/                  # Pipeline verification PNGs
├── docs/
│   ├── PROJECT_OVERVIEW.md         # This file
│   ├── REFACTOR_LOG.md             # Change log
│   ├── implementation plan.md      # Phase 2–3 implementation plan
│   └── archive/                    # Superseded historical docs
├── tests/                          # Test stubs (6 files, all pass)
├── models/                         # (empty) Trained model artifacts
├── configs/                        # (empty) For future YAML configs
└── scripts/                        # (empty) For future utility scripts
```

---

## Key Output Files

| File | Contents |
|------|----------|
| `outputs/eda/statistics.json` | Dataset summary, intensity stats, tumor coverage, class imbalance, morphology, volume stats |
| `outputs/eda/enhanced_volume_stats.json` | Per-split volume stats, HU window recommendations, 2.5D context analysis, outlier report |
| `outputs/preprocessing/preprocessing_config.json` | Final preprocessing parameters for Phase 4 |
| `outputs/preprocessing/volume_index.pkl` | Serialized volume index — load in Phase 4 to avoid re-scanning 58K files |
| `data/splits/{train,val,test}_volumes.txt` | Comma-separated volume IDs per split |
| `data/metadata/phase1_summary.json` | Volume counts, slice counts per split |

---

## Running the Pipeline

```python
# 1. Ensure the project root is on sys.path (notebooks do this automatically)
# 2. Import what you need:
from src.data_loader import DatasetConfig, DataPathManager, create_2d_dataloaders
from src.preprocessing import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.gpu_utils import DEVICE, to_tensor, to_numpy

# 3. Load cached volume index (avoids 58K file scan):
import pickle
with open(DatasetConfig.PREP_OUTPUT_DIR / "volume_index.pkl", "rb") as f:
    volume_index = pickle.load(f)

# 4. Create dataloaders with transforms:
train_loader, val_loader, test_loader = create_2d_dataloaders(
    volume_index, train_vids, val_vids, test_vids,
    batch_size=8,
    transform_train=AugmentedPreprocessingTransform(
        target_size=(256,256), hu_low=-100, hu_high=400,
        clahe=CLAHEProcessor(clip=2.0, grid=(8,8)),
    ),
    transform_val=PreprocessingTransform(
        target_size=(256,256), hu_low=-100, hu_high=400,
    ),
)
```

All paths are dynamically derived from `src/data_loader.py`'s location. No hardcoded machine-specific paths needed.

---

## Hardware Configuration

- **GPU:** NVIDIA RTX 3050 Ti (4GB VRAM)
- **CPU:** Intel Core i7-11800H
- **RAM:** 16GB DDR4
- **Storage:** ~12GB for dataset + ~200MB for project + outputs

The pipeline is tuned for 4GB VRAM:
- Batch size 8 at 256×256 ≈ 1.5GB VRAM for training
- Batch size 4 at 256×256 for evaluation/EDA
- Mixed precision (fp16) enabled in config

---

## Key Decisions

1. **2D PNG pipeline** — All data pre-converted from NIfTI to 2D PNG slices
2. **Binary segmentation** — Masks are 0/1 (liver+tumor combined), not 3-class
3. **Volume-wise splits** — All slices from one volume go to one split (no leakage)
4. **Single-slice input** — 2.5D disabled for now (context=3 is configured but unused)
5. **On-the-fly transforms** — Preprocessing (HU window, CLAHE, resize) applied in dataloader, not pre-computed
6. **GPU-accelerated** — Batch operations on GPU where possible; CPU fallback always available

# Liver Tumor Segmentation & Analysis Platform

> **Current status:** research prototype, not for clinical use. `Practice/` is
> the canonical evidence store for dataset observations and experiment
> decisions. See `docs/DATA_REFERENCE.md` for verified geometry and corrections.

**Hardware**: ASUS TUF F15 · RTX 3050 Ti 4GB · 16GB RAM  
**Framework**: PyTorch 2.x  
**Task**: Binary liver tumor/lesion segmentation from 2D CT PNG slices

---

## Dual-Role Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Liver-CT AI Platform                   │
├─────────────────────────┬────────────────────────────────┤
│   ML/DL Engineer Role   │     Data Analyst Role           │
│   (Modeling & Training) │   (Insights & Reporting)        │
├─────────────────────────┼────────────────────────────────┤
│ • Model Zoo             │ • Tumor Burden Analysis         │
│ • UP³RE-Net Research    │ • Patient Profile Builder       │
│ • Trainer CLI           │ • Clinical Insight Engine        │
│ • Experiment Tracking   │ • Report Generator              │
│ • Hyperparameter Tuning │ • Data Quality Checks           │
│ • Inference API         │ • Interactive Dashboard         │
│ • ONNX/TorchScript      │ • Clinical Knowledge Base       │
└─────────────────────────┴────────────────────────────────┘
```

---

## Quick Start

### Reproducible Windows environment
```powershell
$env:UV_CACHE_DIR = "$PWD/.uv-cache"
uv python install 3.11 --install-dir .uv-python --no-bin --no-registry
uv venv .venv --python .uv-python/cpython-3.11.15-windows-x86_64-none/python.exe
uv pip install --python .venv/Scripts/python.exe torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

### Research validation
```powershell
.venv/Scripts/python.exe tools/analyze.py
.venv/Scripts/python.exe tools/train.py --config configs/experiments/research_baseline.yaml
.venv/Scripts/python.exe tools/evaluate.py --config configs/experiments/research_baseline.yaml --checkpoint models/research_validation/baseline_focal_dice/best_checkpoint.pth
```

### Engineer: Train a model
```bash
python scripts/train.py --model mobilenetv2unet --epochs 50 --batch-size 8
```

### Analyst: Run analysis
```bash
python scripts/analyze.py --mode dataset --format txt
python scripts/analyze.py --mode cohort --splits data/splits
python scripts/analyze.py --mode quality
```

### Analyst: Generate reports
```bash
python scripts/report.py --type dataset
python scripts/report.py --type model --model-name "UPRE-Net" --metrics '{"dice": 0.85, "iou": 0.74}'
python scripts/report.py --type patient --volume-id 5
```

### Analyst: Launch interactive dashboard
```bash
streamlit run app/dashboard.py
```

---

## Pipeline Status

| Phase | Status | Notebook |
|-------|--------|----------|
| 1: Data Loading | ✅ Complete | `notebooks/01_data_loading.ipynb` |
| 2: EDA | ✅ Complete | `notebooks/02_eda.ipynb` |
| 3: Preprocessing | ✅ Complete | `notebooks/03_preprocessing.ipynb` |
| 4: UP³RE-Net Research | 🔧 Implemented | `notebooks/04_research_training.ipynb` |
| 5: Analytics Engine | 🔧 Partial | `src/framework/analytics/` |
| 6: Quality Checks | 🔧 Partial | `src/framework/analytics/quality_checks.py` |
| 7: Testing | ⚠️ Present; environment verification required | `tests/` |
| 8: CLI Tools | ✅ Complete | `scripts/` |
| 9: Dashboard | ✅ Complete | `app/dashboard.py` |
| 10: Clinical Docs | ✅ Complete | `docs/clinical/` |

---

## Project Structure

```
Liver/
├── src/                              # Core library
│   ├── analytics/                    # Analyst role modules
│   │   ├── tumor_burden.py           # Tumor volume, surface area, sphericity
│   │   ├── patient_profile.py        # Per-patient aggregation
│   │   ├── clinical_insights.py      # Clinical risk assessment
│   │   └── report_generator.py       # Multi-format report generation
│   ├── quality/                      # Data quality checks
│   │   └── checks.py                 # Image integrity, drift, split balance
│   ├── models.py                     # MobileNetV2U-Net, Ensemble
│   ├── trainer.py                    # Training with uncertainty guidance
│   ├── preprocessing.py              # HU windowing, CLAHE, transforms
│   ├── data_loader.py                # Dataset, DataLoaders, splitter
│   ├── augmentation.py               # 3D augmentation transforms
│   ├── metrics.py                    # Dice, IoU, calibration error
│   ├── losses.py                     # Dice, Combined, UWACL
│   ├── config.py                     # + YAML config loader
│   ├── gpu_utils.py                  # Device setup, tensor transfer
│   ├── visualization.py              # Plotting functions
│   └── utils.py                      # Logging, seeding, file I/O
├── app/
│   └── dashboard.py                  # Streamlit interactive dashboard
├── scripts/
│   ├── train.py                      # Training CLI
│   ├── analyze.py                    # Analysis CLI
│   └── report.py                     # Report generation CLI
├── configs/                          # YAML experiment configs
│   ├── baseline.yaml
│   ├── upre_net.yaml
│   └── analyst_default.yaml
├── tests/                            # 75 real unit tests
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   ├── test_data_loader.py
│   ├── test_gpu_utils.py
│   └── test_augmentation.py
├── docs/
│   ├── clinical/                     # Clinical knowledge base
│   │   ├── liver_anatomy.md
│   │   ├── ct_scan_basics.md
│   │   └── tumor_types.md
│   └── analytics/                    # Analytics guides
│       ├── interpreting_results.md
│       └── tumor_burden_guide.md
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   └── 04_research_training.ipynb
├── data/                             # Splits, metadata
├── outputs/                          # EDA plots, analysis outputs
├── models/                           # Trained model checkpoints
├── requirements.txt                  # + pyyaml, streamlit
└── README.md
```

---

## ML/DL Engineer Features

| Feature | Location | Description |
|---------|----------|-------------|
| UP³RE-Net | `src/models.py`, `src/trainer.py` | Uncertainty-Propagated Per-Pixel Reweighting Ensemble |
| Model Zoo | `src/models.py:MODEL_REGISTRY` | MobileNetV2U-Net (extensible) |
| UWACL Loss | `src/losses.py` | Uncertainty-Weighted Adaptive Compound Loss |
| YAML Configs | `configs/` | Experiment configuration management |
| Training CLI | `scripts/train.py` | Headless training entry point |
| Mixed Precision | `src/trainer.py` | FP16 training for 4GB VRAM |

## Data Analyst Features

| Feature | Location | Description |
|---------|----------|-------------|
| Tumor Burden Analysis | `src/analytics/tumor_burden.py` | Volume, surface area, sphericity, location |
| Patient Profiles | `src/analytics/patient_profile.py` | Per-volume aggregation + cohort summaries |
| Clinical Insights | `src/analytics/clinical_insights.py` | Risk assessment, tumor characterization |
| Report Generator | `src/analytics/report_generator.py` | TXT/JSON/HTML reports |
| Data Quality | `src/quality/checks.py` | Image integrity, label noise, drift |
| Dashboard | `app/dashboard.py` | Streamlit interactive analysis |
| Clinical Docs | `docs/clinical/` | Liver anatomy, CT basics, tumor types |

---

## Dataset

| Item | Location |
|------|----------|
| Images (58,638 PNGs) | `Dataset/Liver Img Dataset/` |
| Masks (58,638 PNGs) | `Dataset/LiTS_masks/` |
| Source | [LiTS PNG on Kaggle](https://www.kaggle.com/datasets/andrewmvd/lits-png) |

**Key statistics:** 131 volumes, 58,638 slices, 104/13/14 train/val/test split.  
**Practice EDA estimate:** approximately 822:1 background-to-foreground on a
5,000-slice sample. Definitions and sampling must accompany reported ratios.

---

## Testing

```bash
python -m unittest discover tests -v
# Test count and pass status must be regenerated in the active environment.
```

---

## Documentation

| File | Content |
|------|---------|
| `docs/PROJECT_OVERVIEW.md` | Full project reference |
| `docs/RESEARCH_NOVELTY.md` | Patent claims, prior art |
| `docs/clinical/liver_anatomy.md` | Liver anatomy for CT analysis |
| `docs/clinical/ct_scan_basics.md` | CT scanning fundamentals |
| `docs/clinical/tumor_types.md` | Liver tumor classification |
| `docs/analytics/interpreting_results.md` | How to read segmentation metrics |
| `docs/analytics/tumor_burden_guide.md` | Tumor burden analysis guide |

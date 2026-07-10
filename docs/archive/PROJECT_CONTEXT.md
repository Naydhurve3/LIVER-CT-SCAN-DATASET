# Liver Tumor Segmentation Project - Complete Context Document

**Last Updated:** 2026-05-18  
**Project Path:** `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`

---

## 📁 PROJECT STRUCTURE

```
Liver/
├── Phase 1 Data Loading.ipynb          ← COMPLETED ✅
├── MASTER_PLAN.md                      ← 17 notebooks planned
├── EXECUTION_GUIDE.md                 ← Step-by-step guide
├── COMPLETE_WORKFLOW.md               ← Detailed workflow
├── PROJECT_WORKFLOW.md               ← Visual workflow
├── README.md                          ← Project overview
├── Untitled-1.py                      ← Phase 1 standalone script
├── docs/
│   └── FULL_ARCHITECTURE.md
├── data/
│   ├── splits/
│   │   ├── train_volumes.txt          ← Volumes 0-103
│   │   ├── val_volumes.txt            ← Volumes 104-116
│   │   └── test_volumes.txt           ← Volumes 117-130
│   └── metadata/
│       ├── dataset.csv
│       └── statistics.json
└── outputs/
    └── eda/
        └── plots/
```

---

## 📊 DATASET INFORMATION

### Data Sources
| Data | URL |
|------|-----|
| Images | https://www.kaggle.com/datasets/andrewmvd/lits-png |
| Masks | https://www.kaggle.com/datasets/davidrohner/lits-segmentation |

### Dataset Paths
```python
IMAGES_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset")
MASKS_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks")
OUTPUT_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Liver\data")
```

### Dataset Statistics (Phase 1 Results)
| Metric | Value |
|--------|-------|
| Total Images | 58,638 |
| Total Masks | 58,638 |
| Total Volumes | 131 (000-130) |
| Image-Mask Ratio | 1:1 (perfect match) |
| Avg slices/volume | 447 |
| Slices Range | 74-987 |
| Image Size | 512×512 |
| Mask Size | 256×256 |

### Volume-Wise Split (80/10/10)
| Split | Volumes | Volume IDs | Slices |
|-------|---------|------------|--------|
| Train | 104 | 0-103 | 40,667 |
| Val | 13 | 104-116 | ~5,800 |
| Test | 14 | 117-130 | ~5,800 |

---

## 🎯 PHASE 1 RESULTS (COMPLETED)

### Execution Summary
```
============================================================
GPU CONFIGURATION
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
VRAM: 4.29 GB
CUDA: 12.4
============================================================
✅ Directories created/verified:
   Images: D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset
   Masks: D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks
   Output: D:\DATA SCIENCE AND ANALYTICS\Liver\data
🔍 Scanning directories...
   Found 58,638 images
   Found 58,638 masks

📊 Dataset Statistics:
   Total volumes: 131
   Total slices: 58,638
   Volumes with masks: 131
   Avg slices/volume: 447

📋 Sample volumes:
   Volume 000: 75 slices, Mask: True
   Volume 001: 123 slices, Mask: True
   Volume 002: 517 slices, Mask: True
   Volume 003: 534 slices, Mask: True
   Volume 004: 841 slices, Mask: True

📊 Volume Split Configuration:
   train: 104 volumes, 40,667 slices
   val: 13 volumes, ~5,800 slices
   test: 14 volumes, ~5,800 slices

🔍 Data Leakage Check:
   ✅ No data leakage detected!
```

### Tumor Statistics (from Phase 1)
```
📈 Tumor Statistics (n=1000):
   Mean tumor coverage: 0.095%
   Median tumor coverage: 0.000%
   Max tumor coverage: 5.235%
   Slices containing tumor: 126 (12.6%)
```

### Key Files Generated
- `data/splits/train_volumes.txt`
- `data/splits/val_volumes.txt`
- `data/splits/test_volumes.txt`
- `data/metadata/dataset_stats.json`
- `data/split_summary.csv`

---

## 🔧 PHASE 1 CODE STRUCTURE (Reference)

### Classes Implemented
1. **DatasetConfig** - Centralized configuration paths
2. **DataPathManager** - Scans directories, extracts volume IDs, groups slices
3. **VolumeWiseSplitter** - Creates 80/10/10 splits by volume
4. **LiverTumorDataset** - Memory-efficient dataset loader

### Key Code Patterns (from Phase 1 notebook)
```python
# GPU Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Extract volume ID from filename
def extract_volume_id(self, filename: str) -> Optional[int]:
    match = re.match(r"(?:Volume|mask)-(\d+)-\d+\.png", filename)
    return int(match.group(1)) if match else None

# Group by volume
def group_by_volume(self, file_paths: List[Path]) -> Dict[int, List[Path]]:
    groups = defaultdict(list)
    for path in file_paths:
        vol_id = self.extract_volume_id(path.name)
        if vol_id is not None:
            groups[vol_id].append(path)
    return dict(groups)

# Volume-wise split
self.splits = {
    'train': list(range(0, 104)),   # 104 volumes
    'val': list(range(104, 117)),   # 13 volumes
    'test': list(range(117, 131)),  # 14 volumes
}

# Mask binarization
mask = (np.array(mask, dtype=np.float32) > 0).astype(np.float32)
```

---

## 📐 PHASE 2: EDA DESIGN

### 8 EDA Steps

| Step | Name | Purpose |
|------|------|---------|
| 2.1 | Image Statistics | Mean/std intensity, histogram distribution |
| 2.2 | Tumor Statistics | Tumor pixel count, coverage percentage |
| 2.3 | Class Imbalance | Background vs Tumor ratio analysis |
| 2.4 | Volume Analysis | Slices per volume, tumor per volume |
| 2.5 | Visual Inspection | Random samples, volume sequences |
| 2.6 | Morphological Analysis | Connected components, region sizes |
| 2.7 | Generate Report | Save statistics.json, plots |
| 2.8 | Key Insights | Summary and recommendations |

### Phase 2 Expected Outputs
```
outputs/eda/
├── statistics.json              ← Complete statistics
│   ├── image_stats: {mean, std, min, max, histogram}
│   ├── tumor_stats: {coverage_mean, coverage_max, slices_with_tumor}
│   ├── class_balance: {background_%, liver_%, tumor_%}
│   └── volume_stats: {slices_per_volume_dist, tumor_per_volume}
│
├── plots/
│   ├── 01_intensity_histogram.png
│   ├── 02_tumor_coverage_boxplot.png
│   ├── 03_class_imbalance_pie.png
│   ├── 04_slices_per_volume.png
│   ├── 05_random_samples_grid.png
│   ├── 06_volume_sequence.png
│   └── 07_connected_components.png
│
└── eda_report.md               ← Markdown summary
```

### Phase 2 Key Insights (to Discover)
1. **TUMOR COVERAGE IS EXTREMELY LOW**: Mean 0.095%, Max 5.235%
2. **MOST SLICES HAVE NO TUMOR**: 87.4% of slices have 0% tumor coverage
3. **CLASS IMBALANCE IS SEVERE**: ~1000:1 background to tumor ratio
4. **VOLUME VARIATION IS HIGH**: 74-987 slices/volume (13x variation)

### Class Imbalance Implications
| Loss Function | Why Needed |
|---------------|------------|
| BCE alone | ❌ Model predicts "no tumor" everywhere |
| Dice alone | ⚠️ May be unstable during training |
| **BCE + Dice** | ✅ Best approach - combines both strengths |

---

## 🏗️ PHASE 3: PREPROCESSING DESIGN

### HU Windowing (for CT images)
```
CT values are in Hounsfield Units (HU):
- Air: -1000
- Water: 0  
- Soft tissue: 20-100
- Bone: +700 to +3000

For liver/tumor: Window [-100, 400] HU
This range captures soft tissue while filtering out bone and air artifacts.
```

### Preprocessing Pipeline
1. **HU Windowing**: Window [-100, 400] HU
2. **Normalization**: [0, 1] range
3. **Resize**: 512×512 → 256×256
4. **CLAHE** (optional): Contrast enhancement
5. **Augmentation**: Geometric transforms

---

## 🧠 PHASE 4: MODEL ARCHITECTURE DESIGN

### Recommended: MobileNetV2 + U-Net
| Model | Parameters | VRAM | Status |
|-------|------------|------|--------|
| **MobileNetV2** | 3.4M | 0.6GB | ✅ Best fit for RTX 3050 Ti |
| EfficientNetB0 | 5.3M | 0.9GB | ✅ Best accuracy |
| UNet++ | 6M | 1.2GB | ✅ Best tumors |
| Attention U-Net | 9M | 1.0GB | ✅ SE blocks |
| VGG16 | 138M | 3.2GB | ❌ Too large |

### U-Net Architecture
```
INPUT: (256, 256, 1) grayscale CT
├── ENCODER (Contracting)
│   ├── Conv2D(16) + BatchNorm + ReLU + Dropout(0.1)
│   ├── Conv2D(16) + BatchNorm + ReLU
│   ├── MaxPool2D(2)
│   ├── Conv2D(32) + BatchNorm + ReLU + Dropout(0.1)
│   ├── Conv2D(32) + BatchNorm + ReLU
│   ├── MaxPool2D(2)
│   ├── Conv2D(64) + BatchNorm + ReLU + Dropout(0.2)
│   ├── Conv2D(64) + BatchNorm + ReLU
│   ├── MaxPool2D(2)
│   └── Conv2D(128) + BatchNorm + ReLU + Dropout(0.3)
├── BOTTLENECK
│   └── Conv2D(128) + BatchNorm + ReLU + Dropout(0.3)
├── DECODER (Expanding)
│   ├── UpSample2D(2) + Concatenate(skip) + Conv2D(64)
│   ├── UpSample2D(2) + Concatenate(skip) + Conv2D(32)
│   ├── UpSample2D(2) + Concatenate(skip) + Conv2D(16)
│   └── UpSample2D(2) + Concatenate(skip) + Conv2D(16)
└── OUTPUT: Conv2D(1, activation='sigmoid')
```

---

## ⚙️ PHASE 5: TRAINING DESIGN

### Loss Function: BCE + Dice (Combined)
```python
# Combined loss - best for class imbalance
bce_dice_loss = BinaryCrossEntropy(y_true, y_pred) + (1 - DiceCoeff(y_true, y_pred))
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | 8-16 |
| Image Size | 256×256 |
| Optimizer | Adam (lr=1e-4) |
| Loss | BCE (0.5) + Dice (0.5) |
| Epochs | 100 (with early stopping) |
| Patience | 5 |

### Two-Phase Training
1. **Phase 1**: Frozen encoder (20 epochs) - lr=1e-3
2. **Phase 2**: Fine-tune all (80 epochs) - lr=1e-5

---

## 📊 PHASE 6: EVALUATION METRICS

### Primary Metrics
| Metric | Formula | Range | Clinical Relevance |
|--------|---------|-------|-------------------|
| **Dice** | 2×Intersection/(A+B) | [0,1] | ⭐ Primary metric |
| **IoU** | Intersection/Union | [0,1] | Standard |
| **HD95** | 95th percentile Hausdorff | [0,∞) | Boundary accuracy |
| **Sensitivity** | TP/(TP+FN) | [0,1] | Don't miss tumors |
| **Specificity** | TN/(TN+FP) | [0,1] | Reduce false positives |

### Expected Results
| Model | Expected Dice | Training Time |
|-------|--------------|---------------|
| MobileNetV2 | 0.78-0.84 | 4-6 hours |
| EfficientNetB0 | 0.80-0.86 | 6-8 hours |
| Ensemble + TTA | 0.86-0.90 | 8-10 hours |

---

## 🔄 PHASE DEPENDENCY FLOWCHART

```
PHASE 1: Data Loading ✅ (COMPLETE)
    ↓
PHASE 2: EDA (Next to implement)
    ↓
PHASE 3: Preprocessing
    ↓
PHASE 4: Model Architecture
    ↓
PHASE 5: Training
    ↓
PHASE 6: Evaluation
    ↓
PHASE 7: Error Analysis
    ↓
PHASE 8: Pipeline & Documentation
```

---

## ⚠️ KEY DECISIONS MADE

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | PyTorch | Already used in Phase 1 |
| Image Size | 256×256 | Balanced detail/speed |
| Split Method | Volume-wise | No data leakage |
| 2D vs 2.5D | 2D first | Simpler, faster |
| Backbone | MobileNetV2 | Fits RTX 3050 Ti 4GB |
| Primary Metric | Dice | Clinical relevance |

---

## 📋 NOTEBOOK TEMPLATE FORMAT

Each notebook should follow this structure (same as Phase 1):

```python
# %% [markdown]
# # 🏥 PHASE X: [PHASE NAME]
# 
# **Objective:** [What this phase accomplishes]
# 
# **Author:** [Your Name]
# **Date:** 2026-05-18
# **Hardware:** NVIDIA RTX 3050 Ti 4GB

# %% [markdown]
# ## 🔧 Cell 1: Import Libraries & GPU Setup

# %%
# [code]

# %% [markdown]
# ## 📁 Cell 2: Configuration

# %%
# [code]

# %% [markdown]
# ## 🎯 [Cell Name]

# %%
# [code]
```

---

## 🖥️ HARDWARE CONFIGURATION

| Component | Specification |
|-----------|---------------|
| Laptop | ASUS TUF F15 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| VRAM | 4.29 GB |
| RAM | 16 GB |
| CUDA | 12.4 |
| Environment | `ds_gpu` conda environment |
| Python | 3.11.15 |

---

## 📝 LEARNING STRUCTURE (8 PHASES)

| Phase | Name | Time | Key Learning |
|-------|------|------|--------------|
| 1 | Data Loading | 1-2 days | Volume-wise split, file handling |
| 2 | EDA | 2-3 days | Statistics, class imbalance discovery |
| 3 | Preprocessing | 1-2 days | HU windowing, normalization, augmentation |
| 4 | Architecture | 2-3 days | U-Net, backbones, transfer learning |
| 5 | Training | 2-3 days | Loss functions, callbacks, optimization |
| 6 | Evaluation | 1-2 days | Metrics, visualization, comparison |
| 7 | Error Analysis | 1-2 days | Failure modes, improvement strategies |
| 8 | Pipeline | 1-2 days | Integration, documentation |

**Total Estimate:** ~11-17 days (learning pace)

---

## 🔑 CRITICAL INSIGHTS

### Why Volume-Wise Split is CRITICAL
```
❌ WRONG: Random split by slices
   → Same patient's slices in train AND val
   → Model memorizes patients, not generalizes

✅ RIGHT: Split by volume (patient)
   → Each patient appears in only ONE set
   → True test of generalization
```

### Why Class Imbalance Matters
```
Tumor typically occupies only ~0.1% of the image!
That's 50:1 or even 100:1 class imbalance.
→ Standard accuracy metric will be misleading
→ Must use Dice coefficient as primary metric
→ Combined BCE + Dice loss is essential
```

---

## ✅ PHASE 1 VERIFICATION CHECKLIST

- [x] GPU detected: `cuda`
- [x] GPU name: `RTX 3050 Ti`
- [x] Dataset loaded: 58,638 images/masks
- [x] Volume split: 104/13/14 volumes
- [x] Data leakage: VERIFIED NONE
- [x] Splits saved to files
- [x] Metadata generated

---

## 📌 NEXT STEPS

1. **Create Phase 2 EDA notebook** following Phase 1 structure
2. **Load Phase 1 outputs** (splits, metadata) for analysis
3. **Generate comprehensive statistics and visualizations**
4. **Save EDA report** for Phase 3 Preprocessing

---

*End of Context Document*
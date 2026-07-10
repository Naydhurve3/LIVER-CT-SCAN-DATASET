# Liver Tumor Segmentation — Complete Architecture

**Hardware**: ASUS TUF F15 · RTX 3050Ti 4GB · 16GB RAM  
**Dataset**: 58,638 slices · 131 volumes  
**Masks**: ✅ Real LiTS (58,638 PNG)  
**Status**: Research-grade pipeline

---

## Dataset Information

| Item | Value |
|------|-------|
| Images Path | `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset` |
| Masks Path | `D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks` |
| Total Volumes | 131 |
| Total Slices | 58,638 |
| Image Size | 256×256 |

### Data Sources

| Data | Source URL |
|------|-----------|
| Images | https://www.kaggle.com/datasets/andrewmvd/lits-png |
| Masks | https://www.kaggle.com/datasets/davidrohner/lits-segmentation |

### Volume Split (80/10/10)

| Split | Volumes | Count |
|-------|---------|-------|
| Train | 0-103 | 104 (~80%) |
| Val | 104-116 | 13 (~10%) |
| Test | 117-130 | 14 (~10%) |

---

## SECTION 1: MODEL ARCHITECTURES

### 1.1 Model Comparison

| # | Model | Params | VRAM | Batch | Input | Status |
|---|-------|--------|------|-------|-------|--------|
| 1 | mobilenetv2 | 3.4M | 0.6GB | 256×3 | ✅ Best fit |
| 2 | efficientnetb0 | 5.3M | 0.9GB | 256×3 | ✅ Top accuracy |
| 3 | unet_plus_plus | 6M | 1.2GB | 256×3 | ✅ Best tumors |
| 4 | deeplabv3_plus | 5M | 1.5GB | 256×3 | ✅ Multi-scale |
| 5 | attention_unet | 9M | 1.0GB | 256×1 | ✅ SE blocks |
| 6 | hybrid_lite | 6M | 0.8GB | 224×3 | ✅ Fast |
| 7 | dense_unet | 12M | 1.5GB | 256×1 | ⚠️ |
| 8 | resnet50 | 25M | 1.5GB | 256×3 | ✅ |
| 9 | vgg16 | 138M | 3.2GB | 224×3 | ⚠️ Remove |
| 10 | unet_scratch | 7M | 0.8GB | 256×1 | ✅ Baseline |

---

## SECTION 2: TRAINING PROCESS

### 2.1 Pipeline

```
1. AutoConfig → auto-detect masks, derive params
2. Dataset → volume-aware split, preprocessing
3. Two-Phase Training → frozen encoder → fine-tune all
4. Inference → TTA + post-processing
```

### 2.2 Loss Functions

| Loss | Use Case |
|------|----------|
| mixed_bce_dice_boundary | ✅ Real masks (recommended) |
| bce_dice | Standard |
| focal_tversky | Low positive ratio (<2%) |

### 2.3 HW Constraints (RTX 3050Ti 4GB)

| Rule | Value |
|------|-------|
| FP16 | REQUIRED |
| Max batch | 8 (light), 4 (heavy) |
| Image size | 256 |

---

## SECTION 3: FLOWCHARTS

### 3.1 Main Flow

```
START → AutoConfig → Build Dataset → Train (2 phases) → Inference → OUTPUT
```

---

## SECTION 4: SHORT REFERENCE

### 4.1 Quick Start

```python
from src.auto_pipeline import AutoPipeline
pipe = AutoPipeline('mobilenetv2', DATA_DIR).run()
```

### 4.2 Expected Dice (Realistic)

| Model | Dice |
|-------|------|
| MobileNetV2 | 0.78-0.84 |
| EfficientNetB0 | 0.80-0.86 |
| UNet++ | 0.82-0.88 |
| Ensemble + TTA | 0.86-0.90 |

### 4.3 Key Features

- ✅ Real LiTS masks
- ✅ Volume-aware splits
- ✅ HU windowing
- ✅ 2.5D input
- ✅ TTA inference
- ✅ Post-processing
- ✅ K-fold CV
- ✅ Boundary loss

---

## File Structure

```
Liver/
├── src/
│   ├── auto_config.py, auto_pipeline.py
│   ├── dataset.py, dataset_25d.py
│   ├── losses.py, postprocessing.py
│   ├── tta_inference.py, training_utils.py
│   └── models/model_factory.py
├── configs/
│   ├── hardware_profile.yaml
│   ├── dataset_config.yaml
│   ├── experiment_log.yaml
│   └── *.yaml
├── docs/
│   └── FULL_ARCHITECTURE.md
└── README.md
```

---

*Last Updated: 2026-05-10*
# Liver Tumor Segmentation - Complete Workflow

## Overview

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           PROJECT OVERVIEW                                         │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  Task: Semantic Segmentation of Liver Tumors in CT Scans                          │
│  Dataset: LiTS (Liver Tumor Segmentation Challenge)                               │
│  Total Volumes: 131 CT scans → 58,638 slices                                     │
│  Hardware: RTX 3050Ti 4GB VRAM                                                    │
│  Framework: PyTorch                                                               │
│  Expected Dice: 0.78-0.84 (single) / 0.86-0.90 (ensemble)                       │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Dataset Acquisition

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: DATASET DOWNLOAD                                                          │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                           DOWNLOAD SOURCES                                  │   │
│  ├───────────────────────────────┬────────────────────────────────────────────┤   │
│  │  IMAGES                       │  MASKS                                     │   │
│  │  ─────────────────────────    │  ─────────────────────────                │   │
│  │  Kaggle: andrewmvd/lits-png   │  Kaggle: davidrohner/lits-segmentation   │   │
│  │  58,638 PNG slices            │  58,638 PNG masks                        │   │
│  │  512×512 resolution           │  256×256 resolution                      │   │
│  └───────────────────────────────┴────────────────────────────────────────────┘   │
│                                                                                    │
│  └────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                    │
│  FORMAT:                                                                           │
│  • Images: Volume-{vol:03d}-{slice:03d}.png  (e.g., Volume-000-000.png)          │
│  • Masks:  mask-{vol:03d}-{slice:03d}.png     (e.g., mask-000-000.png)           │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: Data Organization

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: FOLDER ORGANIZATION                                                       │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Dataset/                                                                │      │
│  │  ├── Liver Img Dataset/                                                 │      │
│  │  │   ├── Volume-000-000.png  ──┐                                      │      │
│  │  │   ├── Volume-000-001.png  ──┤── 58,638 CT scan slices             │      │
│  │  │   └── ...                  ──┘                                      │      │
│  │  │                                                                  │      │
│  │  └── LiTS_masks/                                                         │      │
│  │      ├── mask-000-000.png  ──┐                                      │      │
│  │      ├── mask-000-001.png  ──┤── 58,638 segmentation masks         │      │
│  │      └── ...                  ──┘                                      │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  KEY POINT: Images and masks must have matching names for pairing               │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 3: Metadata Generation

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: GENERATE METADATA (generate_metadata.py)                                │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  INPUT:  Raw data folders                                                         │
│                                                                                    │
│  PROCESS:                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐      │
│  │  Scan all 131 volumes → For each of 58,638 slices:                     │      │
│  │                                                                          │      │
│  │  1. Extract volume_index (0-130)                                        │      │
│  │  2. Extract slice_index (0-199)                                         │      │
│  │  3. Create image_name, mask_name                                        │      │
│  │  4. Read mask → count tumor pixels                                      │      │
│  │  5. Set has_tumor = 1 if pixels > 0 else 0                            │      │
│  │  6. Assign split: train/val/test based on volume                       │      │
│  │                                                                          │      │
│  │  Write each row to CSV                                                   │      │
│  └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  OUTPUT:                                                                         │
│  ┌─────────────────────────────┬─────────────────────────────────────────────┐    │
│  │  data/metadata/dataset.csv  │  data/metadata/statistics.json            │    │
│  │  ─────────────────────────  │  ─────────────────────────────            │    │
│  │  58,638 rows                │  total_slices: 58,638                     │    │
│  │  Columns: volume_index,     │  tumor_slices: 7,114 (12.1%)             │    │
│  │          slice_index,       │  empty_slices: 51,524 (87.9%)             │    │
│  │          image_name,        │  positive_ratio: 0.121                   │    │
│  │          mask_name,         │  total_tumor_pixels: 4,489,160          │    │
│  │          has_tumor,         │  mean_tumor_pixels: 631                  │    │
│  │          tumor_pixels,      │                                           │    │
│  │          split              │                                           │    │
│  └─────────────────────────────┴─────────────────────────────────────────────┘    │
│                                                                                    │
│  WHY NEEDED?                                                                       │
│  • Provides index for fast data loading                                           │
│  • Stores tumor information for analysis                                          │
│  • Enables volume-wise splitting without leakage                                  │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 4: Dataset Verification

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: VERIFY DATASET (verify_dataset.py)                                       │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  PURPOSE: Ensure data integrity before training                                  │
│                                                                                    │
│  CHECKS PERFORMED:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  1. Directory Exists     → Images folder and Masks folder present?     │      │
│  │  2. File Count           → 58,638 images = 58,638 masks?               │      │
│  │  3. Volume Range         → All 131 volumes (0-130) present?            │      │
│  │  4. Image-Mask Pairing   → Sample 100 pairs, verify matching           │      │
│  │  5. Shape Check          → Images 512×512, Masks 256×256               │      │
│  │  6. Tumor Distribution   → Count tumor vs empty masks                   │      │
│  │  7. Volume Splits        → Verify train/val/test ranges                 │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  OUTPUT:                                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  [✓] PASSED: Dataset verified successfully!                              │      │
│  │       Total images: 58,638                                                │      │
│  │       Total masks: 58,638                                                 │      │
│  │       Tumor samples: X/100 (Y.Y%)                                         │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  IF FAILS: Fix issues before proceeding to training                              │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Volume-Wise Splitting

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: CREATE VOLUME SPLITS                                                     │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                         SPLIT STRATEGY (80/10/10)                        │      │
│  │                                                                          │      │
│  │   TRAIN (80%)       VAL (10%)           TEST (10%)                      │      │
│  │   ┌─────────┐      ┌─────────┐        ┌─────────┐                     │      │
│  │   │Volumes 0-103│    │Volumes 104-116  │Volumes 117-130 │               │      │
│  │   │  104 vols  │    │   13 vols    │    │   14 vols    │               │      │
│  │   │ ~46,000    │    │  ~5,800      │    │  ~5,800      │               │      │
│  │   │  slices    │    │  slices      │    │  slices      │               │      │
│  │   └─────────┘      └─────────┘        └─────────┘                     │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  WHY VOLUME-WISE SPLIT?                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  ❌ WRONG: Random slice split                                           │      │
│  │     → Same patient's slices in train + val                             │      │
│  │     → Model memorizes patient → OVERFITTING                           │      │
│  │                                                                          │      │
│  │  ✅ CORRECT: Volume-wise split                                          │      │
│  │     → All slices from patient in ONE set only                          │      │
│  │     → No leakage → Realistic evaluation                                │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  OUTPUT FILES:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  data/splits/train_volumes.txt   → "0,1,2,...,103"                      │      │
│  │  data/splits/val_volumes.txt     → "104,105,...,116"                    │      │
│  │  data/splits/test_volumes.txt    → "117,118,...,130"                   │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 6: Data Loading & Preprocessing

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: DATA LOADER (dataset_rs.py - ResearchDataset)                           │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  ResearchDataset Class                                                    │      │
│  │  ├── Reads dataset.csv (from Step 3)                                     │      │
│  │  ├── Filters by split (train/val/test)                                   │      │
│  │  ├── Loads images on-demand (not all in memory)                          │      │
│  │  └── Supports both 2D and 2.5D modes                                      │      │
│  │                                                                          │      │
│  │  ResearchDataGenerator (Keras Sequence)                                 │      │
│  │  ├── Batches the data                                                    │      │
│  │  ├── Applies augmentation on-the-fly                                     │      │
│  │  └── Shuffles between epochs                                             │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  PREPROCESSING PIPELINE:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                                                                          │      │
│  │  Input Image (512×512)                                                  │      │
│  │       │                                                                  │      │
│  │       ├─→ RESIZE: cv2.resize(256×256)                                   │      │
│  │       │     interpolation: INTER_CUBIC                                 │      │
│  │       │                                                                  │      │
│  │       ├─→ CLAHE: Contrast Limited Adaptive Histogram Equalization     │      │
│  │       │     clipLimit=2.0, tileGridSize=8×8                           │      │
│  │       │     → Enhances contrast for tumor visibility                  │      │
│  │       │                                                                  │      │
│  │       ├─→ NORMALIZE:                                                    │      │
│  │       │     • MinMax: img/255 (0-1)                                    │      │
│  │       │     • Z-score: (img-mean)/std                                  │      │
│  │       │                                                                  │      │
│  │       └─→ CHANNEL EXPANSION:                                           │      │
│  │             Grayscale → RGB by stacking                                │      │
│  │             (H,W,1) → (H,W,3)                                          │      │
│  │                                                                          │      │
│  │  Output: (256,256,3) float32 tensor                                     │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  MASK PREPROCESSING:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Input Mask (256×256 PNG)                                               │      │
│  │       │                                                                  │      │
│  │       ├─→ RESIZE: cv2.resize(256×256, INTER_NEAREST)                   │      │
│  │       │                                                                  │      │
│  │       └─→ THRESHOLD: (mask > 127).astype(float32)                      │      │
│  │                                                                          │      │
│  │  Output: (256,256,1) binary mask                                        │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  2.5D OPTION:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Instead of single slice, load 3 consecutive:                          │      │
│  │                                                                          │      │
│  │  [slice n-1] [slice n] [slice n+1]  → Concatenate → (256,256,3)       │      │
│  │                                                                          │      │
│  │  Benefits: Captures context from adjacent slices                       │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 7: Data Augmentation

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: AUGMENTATION (Applied during training)                                  │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  WHY AUGMENT?                                                                      │
│  • Increase effective training data                                               │
│  • Improve model generalization                                                    │
│  • Handle class imbalance (only 12% tumor slices)                                  │
│                                                                                    │
│  AUGMENTATIONS APPLIED (on-the-fly):                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                                                                          │      │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │      │
│  │  │ FLIP         │    │ FLIP         │    │ ROTATION    │               │      │
│  │  │ Horizontal   │    │ Vertical     │    │ 90°         │               │      │
│  │  │ (50% prob)   │    │ (50% prob)    │    │              │               │      │
│  │  └──────────────┘    └──────────────┘    └──────────────┘               │      │
│  │                                                                          │      │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │      │
│  │  │ ZOOM         │    │ BRIGHTNESS   │    │ ELASTIC      │               │      │
│  │  │ 1.1x         │    │ ±20%         │    │ Transform    │               │      │
│  │  └──────────────┘    └──────────────┘    └──────────────┘               │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  IMPORTANT: Apply SAME transform to both image AND mask!                         │
│                                                                                    │
│  IMPLEMENTATION:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  class DataGenerator(tf.keras.utils.Sequence):                          │      │
│  │      def __getitem__(self, idx):                                         │      │
│  │          batch_x, batch_y = self.images[idx], self.masks[idx]          │      │
│  │          if self.augment:                                               │      │
│  │              if random() > 0.5:                                        │      │
│  │                  batch_x = np.fliplr(batch_x)                          │      │
│  │                  batch_y = np.fliplr(batch_y)                          │      │
│  │              if random() > 0.5:                                        │      │
│  │                  batch_x = np.flipud(batch_x)                          │      │
│  │                  batch_y = np.flipud(batch_y)                          │      │
│  │          return batch_x, batch_y                                         │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 8: Model Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: BUILD MODEL (MobileNetV2 + UNet)                                         │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                         ARCHITECTURE OVERVIEW                             │      │
│  │                                                                          │      │
│  │   INPUT        ENCODER           DECODER           OUTPUT              │      │
│  │  (256×256×3)  MobileNetV2      UNet-style         (256×256×1)          │      │
│  │      │            │               │                  │                   │      │
│  │      ↓            ↓               ↓                  ↓                   │      │
│  │   ┌────┐     ┌────────┐     ┌────────┐         ┌────────┐              │      │
│  │   │Conv│     │Pretrain │     │Up+Conv+│         │  Sigmoid│              │      │
│  │   │    │     │Encoder  │     │Concat   │         │         │              │      │
│  │   └────┘     └────────┘     └────────┘         └────────┘              │      │
│  │                                                                          │      │
│  │        3.4M params      ~3.0M params         ~0.4M params               │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  ENCODER (MobileNetV2 - Pretrained on ImageNet):                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Layer Name              Output Size    Skip Connection                 │      │
│  │  ───────────────────────────────────────────────────────────────────────  │      │
│  │  input                  256×256×3                                        │      │
│  │  block1_expand_relu     128×128×64      ✓ (skip1)                       │      │
│  │  block2_expand_relu     64×64×32        ✓ (skip2)                       │      │
│  │  block3_expand_relu     32×32×32        ✓ (skip3)                       │      │
│  │  block4_expand_relu     16×16×96        ✓ (skip4)                       │      │
│  │  block6_expand_relu     8×8×320        ✓ (skip5)                        │      │
│  │  out                   8×8×1280                                      │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  DECODER (UNet-style):                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                                                                          │      │
│  │      8×8 (1280)                                                         │      │
│  │         │                                                               │      │
│  │         ↓ UpSampling2D(2) + Conv2D(256) + Concatenate(skip5)           │      │
│  │      16×16 (256)                                                        │      │
│  │         │                                                               │      │
│  │         ↓ UpSampling2D(2) + Conv2D(256) + Concatenate(skip4)           │      │
│  │      32×32 (128)                                                        │      │
│  │         │                                                               │      │
│  │         ↓ UpSampling2D(2) + Conv2D(128) + Concatenate(skip3)           │      │
│  │      64×64 (64)                                                         │      │
│  │         │                                                               │      │
│  │         ↓ UpSampling2D(2) + Conv2D(64)  + Concatenate(skip2)           │      │
│  │     128×128 (64)                                                        │      │
│  │         │                                                               │      │
│  │         ↓ UpSampling2D(2) + Conv2D(32)                                 │      │
│  │     256×256 (32)                                                        │      │
│  │         │                                                               │      │
│  │         ↓ Conv2D(1, activation='sigmoid')                             │      │
│  │     256×256 (1)  → OUTPUT                                              │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  MODEL SPECS:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Total Parameters: 3.4M                                                 │      │
│  │  Expected VRAM: ~0.6GB (batch_size=8, FP16)                            │      │
│  │  Framework: PyTorch                                                       │      │
│  │  Input: (batch, 3, 256, 256)                                            │      │
│  │  Output: (batch, 1, 256, 256) probabilities 0-1                         │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 9: Training Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 9: TRAINING (train_mobilenetv2.py)                                         │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  TWO-PHASE TRAINING:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                                                                          │      │
│  │  PHASE 1: FROZEN ENCODER (First 20 epochs)                              │      │
│  │  ─────────────────────────────────────────────                           │      │
│  │                                                                          │      │
│  │  • Load pretrained MobileNetV2 (ImageNet weights)                      │      │
│  │  • Freeze all encoder layers (NOT trainable)                            │      │
│  │  • Train only decoder layers                                            │      │
│  │  • Learning Rate: 1e-3                                                  │      │
│  │  • Goal: Let decoder learn segmentation without destroying              │      │
│  │           pretrained features                                          │      │
│  │                                                                          │      │
│  │  Layer Weights:                                                         │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │      │
│  │  │  Encoder (MobileNetV2)  │ Decoder (UNet)  │ Trainable?          │    │      │
│  │  ├─────────────────────────┼─────────────────┼─────────────────────┤    │      │
│  │  │   All layers           │   All layers    │   ✗ (frozen)        │    │      │
│  │  └─────────────────────────┴─────────────────┴─────────────────────┘    │      │
│  │                                                                          │      │
│  │  After Phase 1: Model has basic segmentation ability                   │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                              │                                                     │
│                              ↓                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                                                                          │      │
│  │  PHASE 2: FINE-TUNING (Next 80 epochs)                                 │      │
│  │  ─────────────────────────────────────────────                          │      │
│  │                                                                          │      │
│  │  • Unfreeze ALL layers (encoder + decoder)                             │      │
│  │  • Learning Rate: 1e-5 (10x lower - careful update)                   │      │
│  │  • Goal: Fine-tune entire network for better performance              │      │
│  │                                                                          │      │
│  │  Layer Weights:                                                         │      │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │      │
│  │  │  Encoder (MobileNetV2)  │ Decoder (UNet)  │ Trainable?          │    │      │
│  │  ├─────────────────────────┼─────────────────┼─────────────────────┤    │      │
│  │  │   All layers           │   All layers    │   ✓ (trainable)     │    │      │
│  │  └─────────────────────────┴─────────────────┴─────────────────────┘    │      │
│  │                                                                          │      │
│  │  After Phase 2: Best model quality                                      │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  LOSS FUNCTION:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  BCE + Dice Loss (alpha = 0.5)                                         │      │
│  │                                                                          │      │
│  │  bce_dice_loss = BinaryCrossEntropy(y_true, y_pred)                   │      │
│  │                 + (1 - DiceCoeff(y_true, y_pred))                       │      │
│  │                                                                          │      │
│  │  Why Combined?                                                          │      │
│  │  • BCE: Helps with pixel-level accuracy                                │      │
│  │  • Dice: Optimizes for overlap (our main metric)                       │      │
│  │  • Balanced: Neither dominates                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  OPTIMIZER:                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Adam Optimizer                                                          │      │
│  │  • Phase 1 LR: 1.0e-3                                                   │      │
│  │  • Phase 2 LR: 1.0e-5                                                   │      │
│  │  • Weight Decay: 1.0e-5                                                 │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  CALLBACKS:                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  1. EarlyStopping: patience=10, monitor='val_dice_coef', mode='max'  │      │
│  │  2. ModelCheckpoint: save_best_only=True, monitor='val_dice_coef'     │      │
│  │  3. ReduceLROnPlateau: patience=5, factor=0.5, min_lr=1e-7           │      │
│  │  4. CSVLogger: Save training history                                    │      │
│  │  5. TensorBoard: For visualization                                       │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  HARDWARE OPTIMIZATION:                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  • Mixed Precision (FP16) - Reduces VRAM by ~50%                     │      │
│  │  • Batch Size: 8 (safe for 4GB GPU)                                    │      │
│  │  • Memory Growth: True (allocate as needed)                             │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  SAVED OUTPUTS:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  outputs/mobilenetv2/                                                    │      │
│  │  ├── best_model.pth       ← Best validation Dice model                 │      │
│  │  ├── metrics.json        ← Training statistics                         │      │
│  │  └── training_curves.png ← Loss/Dice plots                             │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 10: Inference

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 10: INFERENCE                                                               │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  Load trained model → Process test images → Get predictions                       │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  INFERENCE PIPELINE:                                                      │      │
│  │                                                                          │      │
│  │  Test Images (from data/splits/test_volumes.txt)                        │      │
│  │       │                                                                  │      │
│  │       ↓ (same preprocessing as training)                                │      │
│  │  1. Load & Preprocess                                                    │      │
│  │     • Resize to 256×256                                                  │      │
│  │     • Apply CLAHE                                                        │      │
│  │     • Normalize                                                          │      │
│  │     • Convert to RGB                                                    │      │
│  │       │                                                                  │      │
│  │       ↓                                                                  │      │
│  │  2. Model Prediction                                                    │      │
│  │     • batch_size = 8                                                     │      │
│  │     • Forward pass                                                       │      │
│  │     • Output: (batch, 1, 256, 256) probabilities 0-1                │      │
│  │       │                                                                  │      │
│  │       ↓                                                                  │      │
│  │  3. Collect predictions                                                 │      │
│  │     • Store all test predictions                                        │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  KEY POINT: Model outputs PROBABILITIES (0-1), not binary masks!                │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 11: Test-Time Augmentation (TTA)

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 11: TTA (Test-Time Augmentation) - tta_inference.py                        │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  WHAT IS TTA?                                                                      │
│  Apply multiple augmentations at inference time, average predictions             │
│  → Better predictions without retraining!                                        │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  TTA FLOW (4 augmentations):                                            │      │
│  │                                                                          │      │
│  │     Input Image                                                          │      │
│  │         │                                                               │      │
│  │    ┌─────┴─────┬─────────┬─────────┐                                   │      │
│  │    ↓           ↓         ↓         ↓                                    │      │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                                │      │
│  │  │ Orig │  │H-Flip│  │V-Flip│  │H+V   │                                │      │
│  │  │      │  │      │  │      │  │Flip  │                                │      │
│  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                                │      │
│  │     │         │         │         │                                      │      │
│  │     ↓         ↓         ↓         ↓                                      │      │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                                │      │
│  │  │Pred  │  │Pred  │  │Pred  │  │Pred  │                                │      │
│  │  │      │  │→Rev  │  │→Rev  │  │→Rev  │                                │      │
│  │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘                                │      │
│  │     │         │         │         │                                      │      │
│  │     └─────────┴────┬────┴─────────┘                                      │      │
│  │                    ↓                                                      │      │
│  │              ┌─────────────┐                                               │      │
│  │              │   AVERAGE   │  ← Mean of 4 predictions                   │      │
│  │              └──────┬──────┘                                               │      │
│  │                     ↓                                                      │      │
│  │           Enhanced Prediction                                              │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  AUGMENTATIONS:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  1. Original                                                             │      │
│  │  2. Horizontal flip + reverse                                           │      │
│  │  3. Vertical flip + reverse                                              │      │
│  │  4. Both flips + reverse                                                │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  IMPROVEMENT: +2-3% Dice score                                                    │
│                                                                                    │
│  IMPLEMENTATION:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  def tta_predict(model, images, num_augmentations=4):                  │      │
│  │      predictions = []                                                   │      │
│  │      # Original                                                          │      │
│  │      predictions.append(model(images))                                   │      │
│  │      # H-flip                                                           │      │
│  │      pred = model(tf.reverse(images, axis=2))                         │      │
│  │      predictions.append(tf.reverse(pred, axis=2))                     │      │
│  │      # ... similarly for v-flip and both                                │      │
│  │      return tf.reduce_mean(predictions, axis=0)                        │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 12: Post-Processing

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 12: POST-PROCESSING - postprocessing.py                                      │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  WHY POST-PROCESSING?                                                              │
│  • Raw predictions have noise and false positives                                │
│  • Clean up small components                                                      │
│  • Improve segmentation quality                                                   │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  POST-PROCESSING PIPELINE:                                               │      │
│  │                                                                          │      │
│  │  TTA Output (probabilities)                                              │      │
│  │       │                                                                  │      │
│  │       ↓                                                                  │      │
│  │  1. THRESHOLD                                                            │      │
│  │     (prob > 0.5) → binary mask                                          │      │
│  │       │                                                                  │      │
│  │       ↓                                                                  │      │
│  │  2. CONNECTED COMPONENT FILTERING                                       │      │
│  │     Remove components < min_area (default: 50 pixels)                  │      │
│  │     → Removes small false positives                                     │      │
│  │       │                                                                  │      │
│  │       ↓                                                                  │      │
│  │  3. MORPHOLOGICAL CLEANUP                                               │      │
│  │     • MORPH_CLOSE: Fill small holes                                     │      │
│  │     • MORPH_OPEN: Remove small noise                                    │      │
│  │     • Kernel: 5×5 ellipse                                               │      │
│  │       │                                                                  │      │
│  │       ↓                                                                  │      │
│  │  4. HOLE FILLING                                                         │      │
│  │     Fill internal holes < 50% of mask area                             │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  APPLY POST-PROCESSING:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  from src.postprocessing import apply_postprocessing                    │      │
│  │                                                                          │      │
│  │  for pred in predictions:                                               │      │
│  │      cleaned = apply_postprocessing(pred, min_area=50)                 │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 13: Evaluation

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 13: EVALUATION                                                              │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  CALCULATE METRICS:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │                                                                          │      │
│  │  METRICS COMPUTED:                                                      │      │
│  │  ─────────────────────                                                  │      │
│  │                                                                          │      │
│  │  1. DICE COEFFICIENT (Primary Metric)                                   │      │
│  │     Formula: 2 × |intersection| / (|true| + |pred|)                    │      │
│  │     Range: 0-1, Higher is better                                        │      │
│  │     Meaning: Overlap between predicted and ground truth                 │      │
│  │                                                                          │      │
│  │  2. IOU / JACCARD INDEX                                                │      │
│  │     Formula: |intersection| / |union|                                  │      │
│  │     Range: 0-1, Lower than Dice                                         │      │
│  │     Meaning: Union-based overlap measure                                │      │
│  │                                                                          │      │
│  │  3. SENSITIVITY (Recall / True Positive Rate)                          │      │
│  │     Formula: TP / (TP + FN)                                            │      │
│  │     Range: 0-1, Higher is better                                        │      │
│  │     Meaning: How many tumor pixels are correctly identified            │      │
│  │     → CRITICAL for medical imaging!                                     │      │
│  │                                                                          │      │
│  │  4. SPECIFICITY (True Negative Rate)                                   │      │
│  │     Formula: TN / (TN + FP)                                            │      │
│  │     Range: 0-1, Higher is better                                        │      │
│  │     Meaning: How many background pixels are correctly identified       │      │
│  │                                                                          │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  EXPECTED RESULTS:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  Model              │  Val Dice  │  Test Dice  │  Training Time        │      │
│  │  ───────────────────┼────────────┼─────────────┼────────────────────────│      │
│  │  MobileNetV2        │  0.78-0.84 │  0.76-0.82  │  4-6 hours            │      │
│  │  EfficientNetB0     │  0.80-0.86 │  0.78-0.84  │  6-8 hours            │      │
│  │  UNet++            │  0.82-0.88 │  0.80-0.86  │  7-9 hours            │      │
│  │  ───────────────────┼────────────┼─────────────┼────────────────────────│      │
│  │  Ensemble + TTA     │  0.86-0.90 │  0.84-0.88  │  +30 min             │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
│  SAVE RESULTS:                                                                     │
│  ┌──────────────────────────────────────────────────────────────────────────┐      │
│  │  outputs/mobilenetv2/metrics.json                                       │      │
│  │  {                                                                      │      │
│  │    "dice_coef": 0.82,                                                  │      │
│  │    "iou_coef": 0.71,                                                   │      │
│  │    "sensitivity": 0.85,                                                │      │
│  │    "specificity": 0.98                                                 │      │
│  │  }                                                                      │      │
│  └──────────────────────────────────────────────────────────────────────────┘      │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE WORKFLOW                                          │
│                                                                                      │
│                                                                                      │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                      │
│   │   STEP 1    │ ───→ │   STEP 2    │ ───→ │   STEP 3    │                      │
│   │   Download  │      │   Organize  │      │   Metadata  │                      │
│   │   Dataset   │      │   Folders   │      │   (CSV)     │                      │
│   └──────────────┘      └──────────────┘      └──────────────┘                      │
│                                                           │                           │
│                                                           ↓                           │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                      │
│   │   STEP 6    │ ←─── │   STEP 5    │ ←─── │   STEP 4    │                      │
│   │   Data      │      │   Volume    │      │   Verify    │                      │
│   │   Loader    │      │   Splits    │      │   Dataset   │                      │
│   └──────────────┘      └──────────────┘      └──────────────┘                      │
│         │                                                                   │         │
│         ↓                                                                   ↓         │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                      │
│   │   STEP 7    │ ───→ │   STEP 8    │ ───→ │   STEP 9    │                      │
│   │   Augment   │      │   Build     │      │   Train     │                      │
│   │             │      │   Model     │      │   (2-Phase) │                      │
│   └──────────────┘      └──────────────┘      └──────────────┘                      │
│                                                           │                           │
│                                                           ↓                           │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                      │
│   │   STEP 13   │ ←─── │   STEP 12    │ ←─── │   STEP 11    │                      │
│   │   Evaluate  │      │   Post-     │      │   TTA        │                      │
│   │             │      │   Process   │      │              │                      │
│   └──────────────┘      └──────────────┘      └──────────────┘                      │
│                                                         │                             │
│                                                         ↓                             │
│                                                 ┌──────────────┐                      │
│                                                 │   STEP 10    │                      │
│                                                 │   Inference  │                      │
│                                                 └──────────────┘                      │
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          FINAL OUTPUT: Trained Model + Evaluation Metrics  │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Summary

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE SUMMARY                                         │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  PHASE 1: DATA PREPARATION                                                         │
│  ├── Step 1: Download Dataset                                                     │
│  ├── Step 2: Organize Folders                                                     │
│  ├── Step 3: Generate Metadata (CSV)                                              │
│  ├── Step 4: Verify Dataset                                                       │
│  └── Step 5: Volume-wise Splits                                                   │
│                                                                                    │
│  PHASE 2: MODEL BUILDING                                                           │
│  ├── Step 6: Data Loader (ResearchDataset)                                        │
│  ├── Step 7: Augmentation                                                         │
│  ├── Step 8: Build Model (MobileNetV2 + UNet)                                    │
│  └── Step 9: Training (2-Phase)                                                  │
│                                                                                    │
│  PHASE 3: INFERENCE PIPELINE                                                       │
│  ├── Step 10: Inference                                                           │
│  ├── Step 11: TTA (4x augmentations)                                              │
│  ├── Step 12: Post-Processing                                                     │
│  └── Step 13: Evaluation (Dice, IoU, Sensitivity, Specificity)                   │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files Used

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           FILES AND THEIR ROLES                                    │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  GENERATION:                                                                       │
│  ├── generate_metadata.py    → Creates dataset.csv + statistics.json              │
│  └── verify_dataset.py       → Checks data integrity                              │
│                                                                                    │
│  DATA LOADING:                                                                     │
│  ├── dataset_rs.py          → ResearchDataset + ResearchDataGenerator           │
│  └── dataset.py              → Original dataset (alternative)                    │
│                                                                                    │
│  TRAINING:                                                                         │
│  ├── train_mobilenetv2.py   → Main training script                                │
│  ├── auto_pipeline.py       → End-to-end automated pipeline                       │
│  ├── auto_config.py         → Auto configuration                                  │
│  └── model_factory.py       → Model creation                                     │
│                                                                                    │
│  LOSSES & METRICS:                                                                 │
│  ├── losses.py              → All loss functions                                  │
│  └── tta_inference.py      → Test-time augmentation                               │
│                                                                                    │
│  POST-PROCESSING:                                                                  │
│  └── postprocessing.py      → Post-processing utilities                          │
│                                                                                    │
│  CONFIGURATION:                                                                    │
│  ├── configs/*.yaml         → Model/hardware configurations                       │
│  └── data/splits/*.txt     → Volume split definitions                            │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Important Notes

1. **Environment**: Use `ds_gpu` kernel with PyTorch + CUDA
2. **Paths**: Update to your local paths in each script
3. **Memory**: Batch size 8 is safe for 4GB GPU, use FP16
4. **Splits**: Never use random slice split - always volume-wise
5. **Order**: Follow sequence - don't skip verification
6. **TTA**: Adds ~2-3% improvement for free
7. **Post-processing**: Always apply for cleaner results
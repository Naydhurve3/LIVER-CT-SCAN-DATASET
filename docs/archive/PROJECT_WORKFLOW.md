# Liver Tumor Segmentation - Workflow Overview

## Quick Flow (Step by Step)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              START                                          │
│                         Dataset Download                                     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Organize Data                                                      │
│  ───────────────────────                                                     │
│  Download Images + Masks → Place in Dataset folder                         │
│  └──> Dataset/                                                               │
│       ├── Liver Img Dataset/    (58,638 PNG images)                        │
│       └── LiTS_masks/           (58,638 PNG masks)                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Generate Metadata                                                 │
│  ─────────────────────────                                                   │
│  Run: generate_metadata.py                                                  │
│  └──> Creates dataset.csv + statistics.json                                │
│       Links each image ↔ mask + tumor info                                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Verify Dataset                                                    │
│  ───────────────────────                                                    │
│  Run: verify_dataset.py                                                     │
│  └──> Checks: pairing, volumes, corrupt files                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Create Volume Splits                                             │
│  ───────────────────────────                                               │
│  Split into train/val/test (80/10/10) - VOLUME-WISE                       │
│  └──> train_volumes.txt (0-103)                                           │
│       val_volumes.txt   (104-116)                                          │
│       test_volumes.txt  (117-130)                                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Load Data (ResearchDataset)                                       │
│  ─────────────────────────────────────                                      │
│  Use: dataset_rs.py                                                         │
│  └──> Reads CSV → loads images/masks → applies preprocessing              │
│       (resize, CLAHE, normalize, 2.5D optional)                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Build Model                                                       │
│  ───────────────────                                                        │
│  Use: model_factory.py or train_mobilenetv2.py                            │
│  └──> MobileNetV2 + UNet decoder                                           │
│       Pretrained encoder + trainable decoder                               │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: Train (2-Phase)                                                  │
│  ───────────────────────                                                   │
│  Phase 1: Frozen encoder (20 epochs)                                      │
│  Phase 2: Fine-tune all (80 epochs)                                       │
│  └──> Uses: BCE + Dice Loss                                                │
│       Saves: best_model.pth                                                 │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 8: Inference                                                         │
│  ───────────────                                                           │
│  Load trained model → predict on test set                                 │
│  └──> Output: probability masks                                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 9: TTA (Test-Time Augmentation)                          │
│  ──────────────────────────────────────                         │
│  Run: tta_inference.py                                             │
│  └──> Apply flips → predict → average (4 augmentations)         │
│       Improves Dice by 2-3%                                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 10: Post-Processing                                      │
│  ───────────────────────────                                   │
│  Run: postprocessing.py                                         │
│  └──> Threshold → filter small components → morphology → hole fill│
└─────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 11: Evaluate                                        │
│  ───────────────────                                      │
│  Calculate: Dice, IoU, Sensitivity, Specificity                 │
│  └──> Expected: 0.78-0.84 (single model)                        │
│            0.86-0.90 (ensemble + TTA)                          │
└───────────────────────────────────────────────────────────────────┘
                                 │
                                 ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          END                                           │
│                   Trained Model + Results                             │
└───────────────────────────────────────────────────────────────────┘
```

---

## How Steps Connect

```
Dataset Download ─→ Organize Folder ─→ Generate Metadata (CSV)
                                                      │
                                                      ↓
                                              Verify Dataset
                                                      │
                                                      ↓
                                              Create Splits (Volume-wise)
                                                      │
                                                      ↓
                                              ResearchDataset Loader
                                                      │
                                                      ↓
                                              Build Model (MobileNetV2-UNet)
                                                      │
                                                      ↓
                                              Training (Phase 1 + Phase 2)
                                                      │
                                                      ↓
                                              Inference
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    ↓                                   ↓
                               TTA (4x)                           Post-Processing
                                    │                                   │
                                    └─────────────────┴─────────────────┘
                                                      ↓
                                                Evaluation (Metrics)
```

---

## Key Connections

| Step | Connects To | Via |
|------|-------------|-----|
| Download | Organize Folder | Files placed |
| Organize | Generate Metadata | Images/Masks ready |
| Generate Metadata | Verify Dataset | CSV created |
| Verify | Create Splits | Data validated |
| Splits | ResearchDataset | Volume lists |
| ResearchDataset | Build Model | DataLoader |
| Build Model | Training | Model object |
| Training | Inference | Saved .pth |
| Inference | TTA | Raw predictions |
| TTA | Post-Processing | Enhanced predictions |
| Post-Processing | Evaluation | Final masks |

---

## Summary

```
Step 1-4:   DATA PREPARATION     (Download → Organize → Metadata → Verify)
Step 5-7:   MODEL BUILDING       (Load → Build → Train)
Step 8-11:  INFERENCE PIPELINE   (Predict → TTA → PostProcess → Evaluate)
```

Each step depends on the previous one. No step can skip. Flow is linear except TTA and Post-Processing which both receive from Inference and feed to Evaluation.
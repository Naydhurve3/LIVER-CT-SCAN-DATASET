# Liver Tumor Segmentation - Short Workflow

## Step-by-Step Flow

```
Step 1: Download Dataset → Step 2: Organize Folders → Step 3: Generate Metadata
                                                                            ↓
Step 13: Evaluate ← Step 12: Post-Processing ← Step 11: TTA ← Step 10: Inference
                                                                    ↑
Step 9: Training (2-Phase) → Step 8: Build Model → Step 7: Augmentation
                                                            ↑
Step 6: Data Loader ← Step 5: Volume Splits ← Step 4: Verify Dataset
        ↑
Step 3: Generate Metadata (CSV created here)
```

---

## Step Names Only

| Step | Name | Input From | Output To |
|------|------|------------|-----------|
| 1 | Download Dataset | - | Step 2 |
| 2 | Organize Folders | Step 1 | Step 3 |
| 3 | Generate Metadata | Step 2 | Step 4, Step 6 |
| 4 | Verify Dataset | Step 3 | Step 5 |
| 5 | Volume Splits | Step 4 | Step 6 |
| 6 | Data Loader | Step 3, Step 5 | Step 7 |
| 7 | Augmentation | Step 6 | Step 8 |
| 8 | Build Model | Step 7 | Step 9 |
| 9 | Training | Step 8 | Step 10 |
| 10 | Inference | Step 9 | Step 11, Step 12 |
| 11 | TTA | Step 10 | Step 13 |
| 12 | Post-Processing | Step 10 | Step 13 |
| 13 | Evaluate | Step 11, Step 12 | End |

---

## Interconnection Map

```
                    ┌─────────────┐
                    │ Step 1     │
                    │ Download   │
                    │ Dataset    │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 2     │
                    │ Organize   │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 3     │
                    │ Metadata   │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 4     │
                    │ Verify     │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 5     │
                    │ Splits     │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 6     │
                    │ Data Loader│
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 7     │
                    │ Augment    │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 8     │
                    │ Build Model│
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 9     │
                    │ Training   │
                    └──────┬──────┘
                           ↓
                    ┌─────────────┐
                    │ Step 10    │
                    │ Inference  │
                    └──────┬──────┘
                           ├───────────┐
                           ↓           ↓
                    ┌──────────┐  ┌──────────┐
                    │ Step 11  │  │ Step 12  │
                    │   TTA   │  │ Post-Proc│
                    └────┬─────┘  └────┬─────┘
                         └──────┬──────┘
                                ↓
                         ┌─────────────┐
                         │ Step 13    │
                         │ Evaluate   │
                         └──────┬──────┘
                                ↓
                            END
```

---

## Summary

```
PHASE 1: DATA PREPARATION
├── Step 1 → Step 2 → Step 3 → Step 4 → Step 5
└── Download → Organize → Metadata → Verify → Splits

PHASE 2: MODEL BUILDING
├── Step 6 → Step 7 → Step 8 → Step 9
└── Load Data → Augment → Build Model → Train

PHASE 3: INFERENCE PIPELINE
├── Step 10 → (Step 11 + Step 12) → Step 13
└── Predict → TTA/PostProcess → Evaluate
```

---

## Key Connections

- **Step 3 (Metadata)** feeds to both **Step 4** and **Step 6**
- **Step 10 (Inference)** branches to both **Step 11 (TTA)** and **Step 12 (Post-Processing)**
- **Step 11** and **Step 12** both connect to **Step 13 (Evaluate)**
- Everything flows linearly except TTA and Post-Processing which run in parallel after inference
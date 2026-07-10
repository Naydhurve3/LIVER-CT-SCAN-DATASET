# Execution Guide - Updated (Notebook-Based)

## IMPORTANT: Use PyTorch with GPU

**Python Environment:** `ds_gpu`
**Path:** `D:\DATA SCIENCE AND ANALYTICS\ANACONDA\envs\ds_gpu\python.exe`
**GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)

---

## Quick Start

### 1. Verify GPU Works
Open: `notebooks/00_gpu_setup.ipynb`
- Select kernel: `ds_gpu`
- Run all cells
- Expected: `Using: cuda`

### 2. Generate Metadata
Open: `notebooks/01_generate_metadata.ipynb`
- Run all cells
- Expected: `Total slices: 58,638`

### 3. Verify Dataset
Open: `notebooks/02_verify_dataset.ipynb`
- Run all cells
- Expected: `PASSED`

### 4. Smoke Test (Quick)
Open: `notebooks/03_smoke_test_mobilenetv2.ipynb`
- Run all cells
- Epochs: 5
- Expected: Dice improves from 0.3 to 0.7+

### 5. Full Training
Open: `notebooks/04_full_training_mobilenetv2.ipynb`
- Run all cells
- Epochs: 100 (with early stopping)
- Expected: Val Dice 0.78-0.84

---

## Run Order

| Step | Notebook | Purpose | Time |
|------|----------|---------|------|
| 0 | `00_gpu_setup.ipynb` | Verify GPU | 1 min |
| 1 | `01_generate_metadata.ipynb` | Create dataset index | 2 min |
| 2 | `02_verify_dataset.ipynb` | Verify data integrity | 1 min |
| 3 | `03_smoke_test_mobilenetv2.ipynb` | Quick 5-epoch test | 15 min |
| 4 | `04_full_training_mobilenetv2.ipynb` | Full MobileNetV2 | 4-6 hours |
| 5 | `05_full_training_efficientnetb0.ipynb` | EfficientNet baseline | 6-8 hours |
| 6 | `06_full_training_unetpp.ipynb` | UNet++ baseline | 7-9 hours |

---

## How to Open Notebooks

### VS Code
1. Open VS Code
2. Open folder: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`
3. Click on `.ipynb` file
4. Bottom right corner: Select `ds_gpu` kernel
5. Press `Shift+Enter` to run cells

### Jupyter Notebook
1. Open Command Prompt
2. Navigate: `cd D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`
3. Run: `jupyter notebook`
4. Click notebook in browser

---

## Configuration (All Notebooks)

Each notebook has a `CONFIGURATION` cell at the top:

```python
DATA_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset")
MASK_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks")
OUTPUT_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\outputs\model_name")

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
```

**Modify these if your paths are different.**

---

## Expected Metrics

### Smoke Test (5 epochs)
| Epoch | Dice | Loss |
|-------|------|------|
| 1 | 0.30-0.45 | 0.70-0.85 |
| 2 | 0.45-0.60 | 0.50-0.70 |
| 3 | 0.55-0.70 | 0.40-0.55 |
| 4 | 0.65-0.75 | 0.35-0.45 |
| 5 | 0.70-0.78 | 0.30-0.40 |

### Full Training Results
| Model | Expected Dice | Training Time |
|-------|--------------|---------------|
| MobileNetV2 | 0.78-0.84 | 4-6 hours |
| EfficientNetB0 | 0.80-0.86 | 6-8 hours |
| UNet++ | 0.82-0.88 | 7-9 hours |

---

## Red Flags

| Problem | What you see | Solution |
|---------|--------------|----------|
| GPU not found | `Using: cpu` | Select `ds_gpu` kernel |
| OOM | `RuntimeError` | Reduce batch_size to 4 |
| Dice stuck at 0 | Dice always 0 | Check mask path |
| Loss = NaN | `nan` values | Reduce learning rate |
| Very slow | GPU usage low | Check kernel selection |

---

## Output Files

After each notebook, check:

```
Liver\outputs\
├── mobilenetv2\
│   ├── best_model.pth
│   ├── metrics.json
│   └── training_curves.png
├── mobilenetv2_full\
├── efficientnetb0\
└── unetpp\
```

---

## Monitoring GPU

Open new terminal, run:
```bash
nvidia-smi -l 1
```

During training you should see:
- Python process using 3-3.5 GB VRAM
- GPU utilization 60-100%

---

## Checkpoints After Each Step

### Step 0 (GPU Setup)
- [ ] GPU detected: `cuda`
- [ ] GPU name: `RTX 3050 Ti`

### Step 1 (Metadata)
- [ ] CSV created with 58,638 rows
- [ ] Tumor slices: 7,000-8,000

### Step 2 (Verify Dataset)
- [ ] PASSED message shown
- [ ] Pairing errors: 0

### Step 3 (Smoke Test)
- [ ] Dice improved each epoch
- [ ] Loss decreased each epoch
- [ ] No NaN values

### Step 4 (Full Training)
- [ ] Best Val Dice > 0.75
- [ ] `best_model.pth` saved
- [ ] `training_curves.png` generated

---

## Next: Ablation Studies

After full training of all 3 models:

| Notebook | Purpose |
|----------|---------|
| `07_tta_ablation.ipynb` | Test Time Augmentation |
| `08_boundary_loss.ipynb` | Boundary-aware loss |
| `09_25d_vs_2d.ipynb` | 2.5D vs 2D input |
| `10_ensemble.ipynb` | Combine best models |

---

## Common Issues

### "Kernel not found"
- VS Code: Bottom right corner > Select `ds_gpu`
- Restart VS Code if needed

### "Module not found"
- Make sure `ds_gpu` kernel is selected
- Check: Kernel > Change Kernel > ds_gpu

### "Path not found"
- Verify paths at top of notebook
- Use correct paths for your system

---

## Summary

1. Select `ds_gpu` kernel
2. Run GPU setup notebook
3. Run metadata notebook
4. Run verify notebook
5. Run smoke test (5 epochs)
6. Run full training (100 epochs)
7. Repeat for other models

Total time: ~20-25 hours for all 3 models
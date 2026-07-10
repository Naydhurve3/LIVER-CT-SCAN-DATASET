# PyTorch Execution Guide

## Environment

**Python Environment:** `ds_gpu`  
**Path:** `D:\DATA SCIENCE AND ANALYTICS\ANACONDA\envs\ds_gpu\python.exe`  
**GPU:** NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)

---

## Quick Start (PyTorch)

### 1. Verify GPU Works

```bash
cd D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver
conda activate ds_gpu
python test_gpu.py
```

Expected output: `Using: cuda`

### 2. Verify PyTorch Imports

```bash
python test_pytorch.py
python test_imports.py
```

---

## Running Notebooks

### VS Code
1. Open folder: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`
2. Open `.ipynb` file
3. Bottom right: Select `ds_gpu` kernel
4. Press `Shift+Enter` to run cells

### Jupyter Notebook
```bash
conda activate ds_gpu
cd D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver
jupyter notebook
```

---

## Notebook Execution Order

| Step | Notebook | Purpose | Expected Output |
|------|----------|---------|------------------|
| 1 | `00_gpu_setup.ipynb` | Verify GPU + PyTorch | `Using: cuda` |
| 2 | `01_generate_metadata.ipynb` | Create dataset index | `Total slices: 58,638` |
| 3 | `02_verify_dataset.ipynb` | Verify data integrity | `PASSED` |
| 4 | `03_smoke_test_mobilenetv2.ipynb` | Quick 5-epoch test | Dice 0.3 → 0.7+ |
| 5 | `04_full_training_mobilenetv2.ipynb` | Full training (100 epochs) | Val Dice 0.78-0.84 |

---

## Configuration (All Notebooks)

```python
DATA_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset")
MASK_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks")
OUTPUT_DIR = Path(r"D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\outputs\mobilenetv2")

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
```

---

## Training Scripts

### Python Training

```bash
conda activate ds_gpu
python src/train_mobilenetv2.py
```

### Quick Smoke Test

```bash
python src/quick_smoke.py
```

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

---

## Monitoring GPU

```bash
nvidia-smi -l 1
```

During training you should see:
- Python process using 3-3.5 GB VRAM
- GPU utilization 60-100%

---

## Checkpoints

### Step 1 (GPU Setup)
- [ ] GPU detected: `cuda`
- [ ] GPU name: `RTX 3050 Ti`

### Step 2 (Metadata)
- [ ] CSV created with 58,638 rows
- [ ] Tumor slices: 7,000-8,000

### Step 3 (Verify Dataset)
- [ ] PASSED message shown

### Step 4 (Smoke Test)
- [ ] Dice improved each epoch
- [ ] Loss decreased each epoch
- [ ] No NaN values

### Step 5 (Full Training)
- [ ] Best Val Dice > 0.75
- [ ] `best_model.pth` saved

---

## Output Files

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
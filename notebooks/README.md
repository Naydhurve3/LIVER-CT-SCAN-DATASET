# Master Consolidated Jupyter Notebooks

[![Hub](https://img.shields.io/badge/Notebook%20Hub-Consolidated-brightgreen.svg)]()
[![Kernel](https://img.shields.io/badge/Kernel-Python%203.11-blue.svg)]()

## Overview

This directory contains **4 master consolidated Jupyter Notebooks** that unify the entire dataset characterization, spatial forensics, patient-aware split analysis, and model benchmark pipeline.

---

## Consolidated Notebook Index

### 1. [`01_LiTS_Exploratory_Data_Analysis.ipynb`](01_LiTS_Exploratory_Data_Analysis.ipynb)
- **Topic**: Primary Exploratory Data Analysis & Radiometric Intensity Profiling.
- **Key Concepts**: 131 volume cohort overview, slice density, class imbalance ($822:1$), Hounsfield Unit (`[-160, +240] HU`) windowing, and intensity histograms.

### 2. [`02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb`](02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb)
- **Topic**: Data Quality Audit, Spatial Orientation Fixes & ROI Containment.
- **Key Concepts**: Detection of 47 volume $180^\circ$ rotation flips, spatial transform matrices, $4\times$ tumor burden denominator reconciliation, and 2-stage ROI crop containment ($100\%$ containment).

### 3. [`03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb`](03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb)
- **Topic**: Patient-Disjoint Splits, 3D Lesion Morphology & External Protocol.
- **Key Concepts**: Train (104 vols), Val (13 vols), Sealed Test (14 vols) split breakdown, 845 3D connected component quartiles (Q1–Q4), and 3D IRCADb-01 external evaluation protocol.

### 4. [`04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb`](04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb)
- **Topic**: Benchmark Progression, Loss Ablations & Checkpoint Fusion Policy.
- **Key Concepts**: Benchmark evolution from Mark 1 to Mark 4E, 2-stage predicted-liver segmentation architecture, Bounded Recall Loss, and fixed Checkpoint Fusion policy ($	ext{Prob} = \max(P_{\text{control}}, P_{\text{recall\_loss}})$ @ $0.70$ threshold) passing all 6 validation continuation targets.

---

## Execution Guide

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Launch Jupyter Lab / Notebook
jupyter lab
```

---

[Back to Root README](../README.md) | [View Detailed Dataset Reference](../dataset.md)

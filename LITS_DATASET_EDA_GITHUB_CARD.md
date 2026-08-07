# LiTS-17 CT Dataset: Exploratory Data Analysis & Quality Audit Card

[![Dataset](https://img.shields.io/badge/Dataset-LiTS--17%20CT-blue.svg)](https://competitions.codalab.org/competitions/17094)
[![Modality](https://img.shields.io/badge/Modality-3D%20Abdominal%20CT-green.svg)]()
[![Volumes](https://img.shields.io/badge/Volumes-131-orange.svg)]()
[![Slices](https://img.shields.io/badge/Slices-58%2C638-purple.svg)]()
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)]()

> **Project Reference**: Liver CT Scan Segmentation & Quality Assurance Pipeline  
> **Source Directory**: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver` (`mark 1`, `mark 1 (part 2)`, `Practice`)  
> **Canonical Build ID**: `build_corrected_20260713_214847_v2`  
> **Manifest SHA-256**: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`

---

## Executive Summary

This repository presents a **rigorous exploratory data analysis (EDA), data quality audit, and pre-training characterization** of the Liver Tumor Segmentation Challenge (LiTS-17) dataset. 

Medical image segmentation models often fail not due to architectural limitations, but because of hidden dataset artifacts, orientation flips, spatial denominator mismatches, and severe class imbalance. This analysis uncovers and resolves **47 volume orientation flips**, reconciles a **4x tumor burden metric anomaly**, establishes **patient-disjoint splits**, and presents detailed 3D lesion & radiometric profiles to ensure reproducible, clinical-grade model development.

---

## 1. Dataset Overview & Key Metrics

The dataset consists of **131 primary abdominal CT volumes** containing axial slices covering the liver and abdominal cavity.

| Metric | Value | Description / Notes |
| :--- | :---: | :--- |
| **Total Volumes (Patients)** | `131` | Volume IDs `0` through `130` |
| **Total Axial Slices** | `58,638` | Slices across all 131 volumes |
| **Slices per Volume** | `447.6 ± 274.2` | Range: `[74, 987]`, Median: `432` |
| **Native Spatial Resolution** | `512 x 512` | Reconstructed / Standardized to `256 x 256` |
| **Tumor-Positive Slices** | `7,169` (`12.23%`) | Slice-level tumor presence |
| **Zero-Tumor Volumes** | `13 / 131` (`9.92%`) | Patients with no liver lesions |
| **Pixel Imbalance Ratio (BG:FG)** | `822 : 1` | Extreme foreground sparsity |
| **Overall Mean Tumor Burden** | `0.12%` | % of volume occupied by tumor (Max: `2.12%`) |
| **Corrupted / Missing Images** | `0 / 58,638` | `100%` file-level data integrity |

---

## 2. Radiometric Properties & Hounsfield Unit (HU) Windowing

Raw CT values are expressed in **Hounsfield Units (HU)**, where water is $0\text{ HU}$ and air is $-1000\text{ HU}$. Abdominal CT scans require targeted intensity windowing to isolate soft tissue contrast in the liver parenchyma.

```
       [ -1000 HU ] ----------- [ -160 HU ... 240 HU ] ----------- [ +3000 HU ]
           Air                      Broad Window                     Bone / Contrast
                                (Liver Parenchyma & Lesions)
```

### Windowing Strategy
* **Broad Abdominal Window**: `[-160, +240] HU` — Preserves surrounding abdominal structures and liver boundaries.
* **Soft Tissue / Liver Window**: `[0, +200] HU` — Enhances subtle hypodense or hyperdense parenchymal lesions.

### Normalized Intensity Statistics (8-bit Scale [0, 255])
* **All Pixels**: Mean = `44.5`, Std = `84.1`
* **Background Pixels**: Mean = `44.4`, Std = `84.1`
* **Tumor Pixels**: Mean = `109.1`, Std = `102.1` *(Tumors appear hyper-intense post-windowing relative to dark background)*
* **Median Tumor-minus-Liver Contrast**: `-34 HU` (Train), `-44 HU` (Val)
* **Robust Contrast-to-Noise Ratio (CNR)**: `-1.518` (Train), `-1.881` (Val)

---

## 3. Data Integrity, Forensics & Quality Audits

Prior to model training, multi-stage data forensics were executed across all 58,638 slices to detect spatial and alignment anomalies.

### Key Forensic Discoveries & Corrections
1. **In-Plane Spatial Orientation Correction**:
   * **Issue**: Audit revealed **47 volumes** exhibited a 180-degree in-plane orientation discrepancy between slice images and segmentation masks.
   * **Resolution**: Volumes `83–99` and `101–130` were corrected via an exact $180^\circ$ rotation transformation matrix; Volumes `0–82` and `100` remained identity.
2. **4x Denominator Metric Reconciliation**:
   * **Issue**: Legacy EDA scripts calculated tumor burden percentage using a $512 \times 512$ pixel denominator ($262,144\text{ px}$) against $256 \times 256$ native masks ($65,536\text{ px}$), artificially deflating tumor burden metrics by $4\times$.
   * **Resolution**: Recomputed native-mask metrics, adjusting true mean train tumor burden from `0.0251%` to `0.1003%`.
3. **Region of Interest (ROI) Containment Gate**:
   * **Result**: `100%` tumor pixel containment (`152,763 / 152,763` pixels) inside predicted 3D liver bounding boxes (largest connected component + 16px spatial padding).

---

## 4. Patient-Aware Split Composition

To prevent data leakage across adjacent axial slices, splits were partitioned strictly at the **patient volume level**.

| Split | Volumes | Total Slices | Tumor-Positive Slices | Tumor-Positive % | Mean Tumor Burden % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `104` | `40,667` | `4,930` | `12.12%` | `0.1003%` |
| **Validation** | `13` | `10,685` | `1,042` | `9.75%` | `0.0874%` |
| **Test (Locked)** | `14` | `7,286` | `1,197` | `16.43%` | `0.3071%` |
| **Total** | **131** | **58,638** | **7,169** | **12.23%** | **0.1218%** |

> [!NOTE]  
> Validation is a faithful proxy for Train's burden distribution. The Test set contains a higher proportion of severe tumor burdens (`0.3071%`), representing an intentional held-out out-of-distribution evaluation.

---

## 5. Tumor Morphology & 3D Component Analysis

Analysis of **7,169 tumor-positive slices** and **845 3D connected components** reveals extreme lesion size variation.

```
Tumor Slice Size Distribution (Pixels per 256x256 Slice):
Mean:   631 px  |  Median: 234 px  |  Std: 816.7 px  |  Range: [1, 3,764] px

   High Frequency (Small Lesions <250 px) ──┐
                                           ├───► Heavily Right-Skewed
   Low Frequency (Large Lesions >2000 px) ─┘
```

* **Lesion Quartiles (3D Volume)**:
  * **Q1 (Small / Very Small)**: `< 0.20 mL` (Requires high-resolution feature maps)
  * **Q2 (Medium-Small)**: `0.20 – 1.50 mL`
  * **Q3 (Medium-Large)**: `1.50 – 15.0 mL`
  * **Q4 (Large / Massive)**: `> 15.0 mL` (Up to `266.35 mL` in Volume 116)
* **Zero & Low-Burden Patients**: 
  * Combined Zero-Tumor ($0\%$) and Low-Burden ($< 0.05\%$) patients account for **84.6% of Train** and **84.6% of Validation**, requiring patient-aware loss balancing.

---

## 6. Model Development Milestones & Benchmark Progression

The project evolved across multiple structured iteration stages (`Mark 1` through `Mark 4E`):

```mermaid
flowchart LR
    M1[Mark 1: Baseline Diagnostics] --> M2[Mark 2: Liver ROI Feasibility]
    M2 --> M3[Mark 3: 2-Stage Overfit Proof]
    M3 --> M4[Mark 4/4B: Validation Smoke]
    M4 --> M4CD[Mark 4C/4D: Loss Ablation]
    M4CD --> M4E[Mark 4E: Checkpoint Fusion]
```

### Summary of Controlling Benchmark Results (Mark 4E)

To overcome failure on difficult hypodense lesions (e.g., `V104` and `V116`), a **Checkpoint Fusion Policy** was implemented:
$$\text{Fused Probability} = \max(P_{\text{control}}, P_{\text{recall\_loss}})$$

| Evaluation Metric | Target Threshold | Mark 4E Result | Status |
| :--- | :---: | :---: | :---: |
| **Mean Dice (9 Tumor-Positive Val Patients)** | $\ge 0.3329$ | **`0.3771`** | **PASS** |
| **V104 Patient Dice** | $\ge 0.0500$ | **`0.1166`** | **PASS** |
| **V116 Patient Dice** | $\ge 0.0100$ | **`0.0105`** | **PASS** |
| **Q1 Small Lesion Detection Rate** | $\ge 35.0\%$ | **`50.57%`** | **PASS** |
| **Positive Predicted-Empty Rate** | $\le 35.0\%$ | **`27.45%`** | **PASS** |
| **Empty-Slice False Positive Rate** | $\le 20.0\%$ | **`5.55%`** | **PASS** |

---

## 7. External Validation Protocol (3D IRCADb-01)

To evaluate cross-dataset generalization without risking data leakage on the locked test set, the **3D IRCADb-01** public dataset was integrated:
* **20 Anonymized Patients** (`2,827` axial slices).
* Standardized using identical `256 x 256` HU `[-160, 240]` normalization pipeline.
* Serves as an un-tuned external benchmark for out-of-domain clinical validation.

---

## 8. Recommended Repository Structure for GitHub

To share these EDA findings and data quality results on GitHub, organize your repository as follows:

```
├── README.md                              <- Primary Dataset & Project Card (this document)
├── data_card.json                         <- Machine-readable dataset metadata & SHA-256 hashes
├── docs/
│   ├── DATA_REFERENCE.md                  <- Detailed column schemas & transform specs
│   ├── ORIENTATION_FORENSICS.md           <- Details on 180-deg flip corrections
│   └── MODEL_BENCHMARKS.md                <- Mark 1 to Mark 4E metric progression
├── figures/
│   ├── split_comparison.png               <- Train/Val/Test volume & slice breakdown
│   ├── tumor_size_distribution.png        <- Lesion size histogram & right-skew plot
│   ├── preprocessing_pipeline.png         <- HU windowing & normalization pipeline
│   ├── tumor_heatmap.png                  <- Spatial occupancy heatmap of liver lesions
│   └── correlation_heatmap.png            <- Parameter correlations (Volume vs Burden)
├── manifests/
│   └── slice_manifest.csv                 <- Verified slice-level metadata (58,638 rows)
└── notebooks/
    ├── 01_lits_exploratory_data_analysis.ipynb
    ├── 02_spatial_orientation_forensics.ipynb
    └── 03_pretraining_dataset_characterization.ipynb
```

---

## Citation & Acknowledgments

If you use this dataset audit framework or preprocessing pipeline, please cite:

```bibtex
@dataset{lits17_ct_eda_audit2026,
  author = {Nayd Hurve},
  title = {LiTS-17 CT Dataset Quality Audit, Spatial Forensics, and Exploratory Data Analysis Card},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Naydhurve3/LIVER-CT-SCAN-DATASET}}
}
```

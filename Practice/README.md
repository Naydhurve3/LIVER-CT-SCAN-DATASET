# Practice: Historical EDA, Forensics & Baseline Investigations

[![Phase](https://img.shields.io/badge/Research%20Phase-Historical%20Evidence-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Canonical%20Evidence%20Store-blue.svg)]()

## Overview

The `Practice/` directory serves as the **canonical evidence store** for initial dataset exploratory data analysis (EDA), spatial orientation forensics, 2.5D context slicing experiments, and radiometric robustness ablations.

---

## Key Notebooks & Forensic Scripts

1. **[`lits_eda.ipynb`](lits_eda.ipynb)**:  
   * Primary exploratory data analysis notebook covering slice counts, class imbalance ($822:1$), intensity distributions, and tumor morphology.
2. **[`dataset_pairing_forensics.ipynb`](dataset_pairing_forensics.ipynb)** & **[`validation_volume_mask_orientation_forensics.ipynb`](validation_volume_mask_orientation_forensics.ipynb)**:  
   * Spatial orientation audit scripts that uncovered the **47 volume $180^\circ$ in-plane rotation flip anomaly** across Kaggle Part 2 and Hugging Face imports.
3. **[`adjacent_slice_2_5d_context_ablation.ipynb`](adjacent_slice_2_5d_context_ablation.ipynb)**:  
   * Investigated 2.5D multi-slice context stacking ($k=3$ channels: $z-1, z, z+1$).
4. **[`organ_normalized_intensity_robustness_ablation.ipynb`](organ_normalized_intensity_robustness_ablation.ipynb)**:  
   * Radiometric intensity normalization and CLAHE contrast enhancement robustness study.
5. **[`recall_aware_focal_tversky_loss_ablation.ipynb`](recall_aware_focal_tversky_loss_ablation.ipynb)**:  
   * Loss function study comparing Cross-Entropy, Focal Tversky, and Bounded Recall Losses.

---

## Diagnostic Text Audits & Reports

- [`dataset summary.txt`](dataset%20summary.txt): Original dataset reconstruction and validation plan.
- [`eda summary.txt`](eda%20summary.txt): Concise quantitative summary of LiTS EDA parameters.
- [`split_tumor_audit data.txt`](split_tumor_audit%20data.txt): Audit recording tumor burden percentages across Train, Validation, and Test splits.
- [`eda_plots/`](eda_plots/): 13 high-resolution visualization charts generated during initial EDA.

---

[Back to Root README](../README.md) | [See Consolidated EDA Notebook](../notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb)

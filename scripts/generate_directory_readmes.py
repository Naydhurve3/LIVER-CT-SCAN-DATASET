import os

# 1. mark 1/README.md
mark1_readme = """# Mark 1 Research & Benchmark Iteration Pipeline (Mark 1 → Mark 4E)

[![Phase](https://img.shields.io/badge/Research%20Phase-Mark%201%20to%204E-blue.svg)]()
[![Status](https://img.shields.io/badge/Mark%204E-Controlling%20Pass-success.svg)]()

## Overview

This directory contains the primary research, model diagnostics, ROI feasibility studies, loss function ablations, and checkpoint fusion experiments executed on the LiTS-17 CT dataset.

The research evolved across **8 sequential iteration notebooks** (`Mark 1` through `Mark 4E`), concluding with the controlling **Mark 4E Checkpoint Fusion Policy**.

---

## Notebook Execution & Process Flow

```mermaid
flowchart TD
    M1[mark_1_probability_contrast_localization_diagnostic.ipynb] --> M2[mark_2_roi_multiwindow_feasibility.ipynb]
    M2 --> M3[mark_3_two_stage_multiwindow_overfit.ipynb]
    M3 --> M4[mark_4_two_stage_validation_smoke.ipynb]
    M4 --> M4B[mark_4b_roi_probability_diagnostics.ipynb]
    M4B --> M4C[mark_4c_two_channel_recall_ablation.ipynb]
    M4C --> M4D[mark_4d_metric_reconciliation_v116_diagnostic.ipynb]
    M4D --> M4E[mark_4e_checkpoint_fusion_validation.ipynb]
```

### Execution Index & Purpose:

1. **[`mark_1_probability_contrast_localization_diagnostic.ipynb`](mark_1_probability_contrast_localization_diagnostic.ipynb)**:  
   * **Purpose**: Baseline diagnostic evaluating model output probabilities and contrast response.  
   * **Output Directory**: [`mark_1_outputs/`](mark_1_outputs/)  
   * **Key Finding**: Identified probability suppression on small hypodense tumors.

2. **[`mark_2_roi_multiwindow_feasibility.ipynb`](mark_2_roi_multiwindow_feasibility.ipynb)**:  
   * **Purpose**: Evaluated predicted-liver ROI crop bounding boxes and multi-window HU input feasibility.  
   * **Output Directory**: [`mark_2_outputs/`](mark_2_outputs/)  
   * **Key Finding**: Validated that Stage-1 ROI cropping reduces slice search area by ~58% without losing liver tissue.

3. **[`mark_3_two_stage_multiwindow_overfit.ipynb`](mark_3_two_stage_multiwindow_overfit.ipynb)**:  
   * **Purpose**: Deterministic overfit gate on 2-stage ROI segmentation pipeline.  
   * **Output Directory**: [`mark_3_outputs/`](mark_3_outputs/)  
   * **Key Metric**: Achieved hard micro-Dice **`0.900557`** after 17 epochs; **100%** tumor containment.

4. **[`mark_4_two_stage_validation_smoke.ipynb`](mark_4_two_stage_validation_smoke.ipynb)**:  
   * **Purpose**: 5-Epoch validation smoke training for generalization testing.  
   * **Output Directory**: [`mark_4_outputs/`](mark_4_outputs/)  
   * **Key Finding**: High positive predicted-empty rate (`36.85%`) on validation slices.

5. **[`mark_4b_roi_probability_diagnostics.ipynb`](mark_4b_roi_probability_diagnostics.ipynb)**:  
   * **Purpose**: Global probability threshold sweep (0.05–0.70).  
   * **Output Directory**: [`mark_4b_outputs/`](mark_4b_outputs/)  
   * **Key Finding**: Threshold-only tuning could not reduce predicted-empty rate below 35% without triggering high false positive rates.

6. **[`mark_4c_two_channel_recall_ablation.ipynb`](mark_4c_two_channel_recall_ablation.ipynb)**:  
   * **Purpose**: Two-channel broad+liver window input & Recall-Aware Loss ablation.  
   * **Output Directory**: [`mark_4c_outputs/`](mark_4c_outputs/)  
   * **Key Finding**: Two-channel input collapsed on tumor-positive patients; Recall Loss improved recall metrics but damaged volume `V116`.

7. **[`mark_4d_metric_reconciliation_v116_diagnostic.ipynb`](mark_4d_metric_reconciliation_v116_diagnostic.ipynb)**:  
   * **Purpose**: Metric reconciliation across 9 tumor-positive validation patients & `V116` failure diagnosis.  
   * **Output Directory**: [`mark_4d_outputs/`](mark_4d_outputs/)  
   * **Key Finding**: Discovered that all `152,763` tumor pixels of `V116` were inside the ROI; failure was due to hypodense domain response, not ROI clipping.

8. **[`mark_4e_checkpoint_fusion_validation.ipynb`](mark_4e_checkpoint_fusion_validation.ipynb)**:  
   * **Purpose**: Fixed Checkpoint Fusion Policy evaluation: $P_{\\text{fused}} = \\max(P_{\\text{control}}, P_{\\text{recall\\_loss}})$.  
   * **Output Directory**: [`mark_4e_outputs/`](mark_4e_outputs/)  
   * **Key Metric**: **PASSED ALL 6 VALIDATION TARGETS** at threshold `0.70` (Mean Dice `0.3771`, Q1 Detection `50.57%`, FP `5.55%`).

---

## Directory Output Index

- [`mark_1_outputs/`](mark_1_outputs/): Diagnostic probability arrays & baseline statistics.
- [`mark_2_outputs/`](mark_2_outputs/): Multi-window ROI crop manifests & spatial bounding box logs.
- [`mark_3_outputs/`](mark_3_outputs/): Overfit model weights (`broad_1ch_overfit.pth`) & containment logs.
- [`mark_4_outputs/`](mark_4_outputs/): Control model weights (`mark_4_best.pth`) & 5-epoch logs.
- [`mark_4c_outputs/`](mark_4c_outputs/): Recall-loss model weights (`recall_loss_best.pth`).
- [`mark_4d_outputs/`](mark_4d_outputs/): Probability cache arrays (`probability_cache/volume_*.npz`).
- [`mark_4e_outputs/`](mark_4e_outputs/): Final fusion gate JSON report (`mark_4e_gate_result.json`).

---

[Back to Root README](../README.md) | [Next Phase: Mark 1 (Part 2)](../mark%201%20(part%202)/README.md)
"""

# 2. Practice/README.md
practice_readme = """# Practice: Historical EDA, Forensics & Baseline Investigations

[![Phase](https://img.shields.io/badge/Research%20Phase-Historical%20Evidence-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Canonical%20Evidence%20Store-blue.svg)]()

## Overview

The `Practice/` directory serves as the **canonical evidence store** for initial dataset exploratory data analysis (EDA), spatial orientation forensics, 2.5D context slicing experiments, and radiometric robustness ablations.

---

## Key Notebooks & Forensic Scripts

1. **[`lits_eda.ipynb`](lits_eda.ipynb)**:  
   * Primary exploratory data analysis notebook covering slice counts, class imbalance ($822:1$), intensity distributions, and tumor morphology.
2. **[`dataset_pairing_forensics.ipynb`](dataset_pairing_forensics.ipynb)** & **[`validation_volume_mask_orientation_forensics.ipynb`](validation_volume_mask_orientation_forensics.ipynb)**:  
   * Spatial orientation audit scripts that uncovered the **47 volume $180^\\circ$ in-plane rotation flip anomaly** across Kaggle Part 2 and Hugging Face imports.
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
"""

# 3. notebooks/README.md
notebooks_readme = """# Master Consolidated Jupyter Notebooks

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
- **Key Concepts**: Detection of 47 volume $180^\\circ$ rotation flips, spatial transform matrices, $4\\times$ tumor burden denominator reconciliation, and 2-stage ROI crop containment ($100\\%$ containment).

### 3. [`03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb`](03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb)
- **Topic**: Patient-Disjoint Splits, 3D Lesion Morphology & External Protocol.
- **Key Concepts**: Train (104 vols), Val (13 vols), Sealed Test (14 vols) split breakdown, 845 3D connected component quartiles (Q1–Q4), and 3D IRCADb-01 external evaluation protocol.

### 4. [`04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb`](04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb)
- **Topic**: Benchmark Progression, Loss Ablations & Checkpoint Fusion Policy.
- **Key Concepts**: Benchmark evolution from Mark 1 to Mark 4E, 2-stage predicted-liver segmentation architecture, Bounded Recall Loss, and fixed Checkpoint Fusion policy ($\text{Prob} = \\max(P_{\\text{control}}, P_{\\text{recall\\_loss}})$ @ $0.70$ threshold) passing all 6 validation continuation targets.

---

## Execution Guide

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Launch Jupyter Lab / Notebook
jupyter lab
```

---

[Back to Root README](../README.md) | [View Detailed Dataset Reference](../dataset.md)
"""

# 4. understanding the project/README.md
understanding_readme = """# Project Architecture, Metrics & Knowledge Base

[![Docs](https://img.shields.io/badge/Docs-Knowledge%20Base-blue.svg)]()

## Overview

This directory contains the **historical architectural documentation, decision logs, and metric contracts** established during the development of the Liver CT Segmentation Platform.

---

## Document Index

1. [`01_dataset_and_provenance.md`](01_dataset_and_provenance.md):  
   Dataset acquisition history, mirror sources, and canonical build specification.
2. [`02_validation_and_metric_contract.md`](02_validation_and_metric_contract.md):  
   Validation gate targets, Dice loss equations, and evaluation contracts.
3. [`03_experiment_timeline.md`](03_experiment_timeline.md):  
   Chronological timeline of all experimental milestones (Mark 1 through Mark 4E).
4. [`04_parameters_and_techniques.md`](04_parameters_and_techniques.md):  
   Hyperparameter configurations, data augmentation rules, and 2-stage ROI extraction algorithms.
5. [`05_complete_results.md`](05_complete_results.md):  
   Comprehensive quantitative results across all baseline and ablation models.
6. [`06_failures_fixes_and_lessons.md`](06_failures_fixes_and_lessons.md):  
   Failure mode adjudication, hypodense lesion localization diagnostics, and lessons learned.
7. [`07_artifact_and_notebook_map.md`](07_artifact_and_notebook_map.md):  
   Mapping of all project notebooks, scripts, weights, and JSON output files.
8. [`08_current_plan.md`](08_current_plan.md):  
   Continuation roadmap for fusion freeze confirmation and external dataset evaluation.
9. [`CHAT_HANDOFF_SUMMARY.md`](CHAT_HANDOFF_SUMMARY.md):  
   Executive handoff summary of current project state.

---

[Back to Root README](../README.md)
"""

# 5. scripts/README.md
scripts_readme = """# Scripts & Execution Tools

## Overview

This directory contains utility scripts for dataset arrangement, manifest generation, notebook consolidation, and report generation.

---

## Utility Index

- `arrange_dataset.py` / `arrange_dataset.ps1`: Automated pipeline for extracting raw NIfTI volumes and constructing the canonical staging build (`build_corrected_20260713_214847_v2`).
- `generate_consolidated_notebooks.py`: Script to generate the 4 master consolidated Jupyter Notebooks in `notebooks/`.
- `generate_modular_docs.py`: Script to generate the modular markdown documentation hub in `docs/`.

---

[Back to Root README](../README.md)
"""

# 6. src/README.md
src_readme = """# Core Framework Source Code (`src/`)

## Overview

This directory contains the core PyTorch deep learning framework modules for dataset loading, loss functions, model architectures, and evaluation tools.

---

## Package Architecture

```
src/framework/
├── data/
│   ├── manifest_dataset.py        <- Verified Manifest Dataset loader
│   └── samplers.py                <- Patient-aware & lesion-balanced samplers
├── losses/
│   ├── focal_tversky.py           <- Focal Tversky Loss implementation
│   └── stability_bounded_recall.py<- Stability Bounded Recall Loss implementation
├── models/
│   ├── mobilenetv2_unet.py        <- MobileNetV2-UNet 2D/2.5D architecture
│   └── two_stage_roi.py           <- Stage-1 Liver ROI + Stage-2 Tumor Net
└── metrics/
    └── segmentation_metrics.py    <- Micro/Macro Dice, Surface Distance, Q1 Recall
```

---

[Back to Root README](../README.md)
"""

# 7. configs/README.md
configs_readme = """# Experiment & Dataset Configurations

## Overview

Contains YAML configuration files specifying dataset paths, split definitions, preprocessing profiles, hyperparameter choices, and model architectures.

---

## Configuration Index

- `datasets/lits_verified_eda.yaml`: Standard configuration for canonical build `build_corrected_20260713_214847_v2`.
- `splits/`: CSV and TXT files defining volume assignment across Train (104), Validation (13), and Sealed Test (14) sets.

---

[Back to Root README](../README.md)
"""

# 8. app/README.md
app_readme = """# Streamlit Clinical Analytics Dashboard

## Overview

Contains the Streamlit interactive clinical dashboard (`app/dashboard.py`) for visualizing volume CT slices, tumor segmentations, patient burden profiles, and model probability heatmaps.

---

## Running the Dashboard

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Launch Streamlit app
streamlit run app/dashboard.py
```

---

[Back to Root README](../README.md)
"""

readmes = {
    "mark 1/README.md": mark1_readme,
    "Practice/README.md": practice_readme,
    "notebooks/README.md": notebooks_readme,
    "understanding the project/README.md": understanding_readme,
    "scripts/README.md": scripts_readme,
    "src/README.md": src_readme,
    "configs/README.md": configs_readme,
    "app/README.md": app_readme
}

for rel_path, content in readmes.items():
    filepath = os.path.join(rel_path)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created README: {filepath}")

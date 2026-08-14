# Liver CT Segmentation Project — Consolidated Evaluation & Project Data

> **Purpose**: This document consolidates **all data required to understand, reproduce, and evaluate** the Liver CT Segmentation project. It is the single-source reference assembled from the 150+ markdown documentation files, machine-readable JSON registers, and YAML configurations across the repository (`README.md`, `dataset.md`, `docs/`, `understanding the project/`, `mark 1/`, `mark 1 (part 2)/`, `Practice/`, `configs/`, `results/`).
>
> **Last consolidated**: 9 August 2026

---

## Table of Contents

1. [Project Identity & Executive Summary](#1-project-identity--executive-summary)
2. [Dataset Provenance & Canonical Build](#2-dataset-provenance--canonical-build)
3. [Cohort Statistics & Patient-Disjoint Splits](#3-cohort-statistics--patient-disjoint-splits)
4. [Slice Manifest Schema](#4-slice-manifest-schema)
5. [Spatial Geometry & Physical Dimensions](#5-spatial-geometry--physical-dimensions)
6. [Radiometrics & HU Windowing](#6-radiometrics--hu-windowing)
7. [3D Lesion Morphology & Stratification Bins](#7-3d-lesion-morphology--stratification-bins)
8. [Data Forensics: Fixes Applied](#8-data-forensics-fixes-applied)
9. [Validation & Metric Contract](#9-validation--metric-contract)
10. [Model Benchmarks: Mark 1 → Mark 4E](#10-model-benchmarks-mark-1--mark-4e)
11. [Complete Experiment Results (Practice Ablations)](#11-complete-experiment-results-practice-ablations)
12. [Multi-Task Model: Epoch Trajectory & Patient Detail](#12-multi-task-model-epoch-trajectory--patient-detail)
13. [One-Time Locked Test Evaluation (Step 04)](#13-one-time-locked-test-evaluation-step-04)
14. [External Evaluation: 3D-IRCADb-01 (Steps 12–21)](#14-external-evaluation-3d-ircadb-01-steps-1221)
15. [Part 2 Research Pipeline Status (Steps 00–21)](#15-part-2-research-pipeline-status-steps-0021)
16. [Model Improvement Program (Step 00)](#16-model-improvement-program-step-00)
17. [Configurations & Framework Parameters](#17-configurations--framework-parameters)
18. [Cryptographic Hashes & Verification Fingerprints](#18-cryptographic-hashes--verification-fingerprints)
19. [Artifact & Notebook Map](#19-artifact--notebook-map)
20. [Licensing](#20-licensing)
21. [Current Status, Blockers & Next Actions](#21-current-status-blockers--next-actions)
22. [UP³RE-Net / MedSegX Research Track (Novelty & Findings)](#22-upre-net--medsegx-research-track-novelty--findings)
23. [MedSegX Research Changelog — Sprint Results & Ablations](#23-medsegx-research-changelog--sprint-results--ablations)
24. [Limitations & Failure Analysis (Final Test)](#24-limitations--failure-analysis-final-test)
25. [Reproducibility Contract & Additional Hashes](#25-reproducibility-contract--additional-hashes)
26. [Result Levels & Success-Criteria Contract](#26-result-levels--success-criteria-contract)
27. [Environment Snapshot](#27-environment-snapshot)
28. [Future Work & Research Directions](#28-future-work--research-directions)

---

## 1. Project Identity & Executive Summary

| Attribute | Value |
| :--- | :--- |
| **Project Name** | Liver CT Scan Dataset & Quality Assurance Platform (LiTS-17) |
| **Repository Purpose** | Rigorous EDA, Spatial Forensics, Pre-training Characterization, 2-Stage Liver Tumor Segmentation Benchmarks |
| **Modality** | 3D Abdominal CT |
| **Primary Dataset** | LiTS-17 (Liver Tumor Segmentation Challenge, CodaLab ID `17094`) |
| **Benchmark Publication** | Bilic P, et al. *The Liver Tumor Segmentation Benchmark (LiTS)*. Medical Image Analysis. 2023;84:102680. doi:[10.1016/j.media.2022.102680](https://doi.org/10.1016/j.media.2022.102680) |
| **Author** | Nayd Hurve |
| **GitHub** | https://github.com/Naydhurve3/LIVER-CT-SCAN-DATASET |
| **Project Root** | `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver` |
| **Model Target** | Binary tumor-only segmentation (liver/organ mask stored separately) |
| **Primary Architecture** | MobileNetV2-U-Net (PyTorch) |

### Key Achievements (Verified)

- **Spatial Alignment Repair**: 47 volumes with 180° in-plane orientation flips corrected (Volumes `83–99`, `101–130`).
- **Metric Reconciliation**: 4× tumor-burden denominator error corrected; true mean train tumor burden `0.1003%`.
- **100% ROI Containment**: 2-stage bounding-box crop pipeline achieves 100% tumor pixel containment (`152,763 / 152,763` pixels, V116).
- **Mark 4E Checkpoint Fusion**: pixelwise maximum fusion (`max(P_control, P_recall_loss)` @ threshold `0.70`) **passed all 6 validation targets**.
- **3D-IRCADb-01 External Evaluation**: frozen external contract **passed** (global Dice `0.847977`) with documented caveats.
- **One-Time Locked Test Evaluation**: global Dice `0.767696`; **formal model acceptance failed** (V121 minimum-patient floor = 0).

---

## 2. Dataset Provenance & Canonical Build

### 2.1 Identity

- **Dataset Name**: Liver Tumor Segmentation Benchmark (LiTS-17) — 3D Abdominal CT Volumes with Liver & Tumour Annotations.
- **Local Cohort**: **131 annotated primary CT volumes**, numbered `0` through `130`.
- **Canonical Build ID**: `build_corrected_20260713_214847_v2`
- **Build Directory**: `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- **Manifest**: `manifests\slice_manifest.csv` (58,638 rows × 27 columns)
- **Manifest SHA-256**: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- **Voxel Annotation Values**: `0 = Background`, `1 = Liver Parenchyma`, `2 = Tumour/Lesion`
- **Sealed Test Set**: internal 14-volume holdout carved from the 131 annotated volumes (**not** the official 70-volume unlabelled LiTS challenge test set).

### 2.2 Source Acquisition Record

| Source Mirror | Volume ID Range | Volume Count | Raw Format |
| :--- | :---: | :---: | :--- |
| Kaggle `andrewmvd/liver-tumor-segmentation` | `0–50` | 51 | NIfTI (`.nii`) pairs |
| Kaggle `andrewmvd/liver-tumor-segmentation-part-2` | `51–69`, `100` | 20 | NIfTI (`.nii`) pairs |
| Hugging Face CADS `0004_lits` | `70–99`, `101–130` | 60 | `.nii.gz` decompressed to `.nii` |
| **Total Primary Cohort** | **`0–130`** | **131** | 131 CT volumes + 131 multiclass segmentations |

### 2.3 Historical Image Working Copies (from `source_registry.json`, SHA-256 `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173`)

| Source Folder | Absolute Local Path | Contents |
| :--- | :--- | :--- |
| `lits-png` | `D:\DATA SCIENCE AND ANALYTICS\Dataset\lits-png\dataset_6\dataset_6` | 58,638 image PNGs, 15,868 liver masks, 15,817 lesion masks |
| `LiTS_masks` | `D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks` | 58,638 tumor-mask PNGs (256×256) |
| `Liver Img Dataset` | `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset` | 58,638 legacy image PNGs (512×512, unverified alignment) |

> These folders are historical extractions of the **same 131-volume cohort**. The canonical build pipeline is the sole source of truth.

### 2.4 Storage Root Layout (`D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\`)

```
├── 00_source_registry/           <- source_registry.json inventory
├── 01_raw_authoritative/         <- volume-0.nii … volume-130.nii + segmentation-0.nii … (131 each)
├── 02_staging/build_corrected_20260713_214847_v2/  <- CANONICAL BUILD
│   ├── images/                   <- 131 volume subdirs, 58,638 PNGs (256×256)
│   ├── organ_masks/              <- 58,638 binary organ PNGs
│   ├── tumor_masks/              <- 58,638 binary tumor PNGs
│   ├── manifests/                <- slice_manifest.csv, eda_*_manifest.csv, quarantine_manifest.csv
│   ├── splits/                   <- train/val/test CSV + volume TXT + split_hashes.json
│   ├── audits/                   <- strict_validation_summary.json, volume_summary.csv, split_summary.csv
│   ├── spatial_reviews/          <- 131 per-volume review sheets + 7 batch contact sheets
│   ├── dataset_version.json / dataset_readiness.json / dataset_card.txt
├── 03_derived_256/               <- 256×256 pre-cropped ROI & cache artifacts
├── 04_manifests/ 05_splits/ 06_audits/ 07_training_cache/ 99_quarantine/
```

### 2.5 Corrected Build Data-Card Facts (Step 01, 2026-08-03)

- Generated: `2026-08-03T08:52:34.580386+00:00`
- Status: `verified_eda_ready`; `training_ready: false` (gated on loader/overfit verification)
- Gate decision: `PASS_FREEZE_DATA_CARD_AND_PROCEED_TO_FUSION_CONFIRMATION`
- Sampling policy: `uniform_patient_aware_existing_sampler_no_change`, seed 42, train-derived bins
- Alignment: 117/117 train+val volumes resolved; volumes 48–52 documented review cases (manual header approval)

---

## 3. Cohort Statistics & Patient-Disjoint Splits

### 3.1 Global Cohort Metrics

| Metric | Value |
| :--- | :---: |
| Total CT Volumes (Patients) | `131` |
| Total Axial Slices | `58,638` |
| Organ-Positive Slices | `19,156` (`32.67%`) |
| Tumour-Positive Slices | `7,169` (`12.23%`) |
| Organ Pixel Count | `87,806,713` |
| Tumour Pixel Count | `4,615,891` |
| Pixel Imbalance Ratio (BG:FG) | `822 : 1` (0.12% tumor pixels) |
| Zero-Tumor Volumes | `13 / 131` (`9.92%`) |
| Slices per Volume | `447.6 ± 274.2`, range `[74, 987]`, median `432` |
| Native Resolution | `512 × 512` → standardized `256 × 256` |
| Data Integrity Rate | `100%` (0 corrupted across 175,914 PNGs) |
| In-Plane Orientation Fixes | 47 volumes `rot180`; 84 volumes `identity` |

### 3.2 Canonical Patient-Disjoint Splits (Authoritative)

| Split | Volume ID Range | Volumes | Total Slices | Organ-Positive | Tumor-Positive Slices | Tumor-Positive % | Mean Tumor Burden % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `0 – 103` | `104` | `40,667` | `13,382` | `4,930` | `12.12%` | `0.1003%` |
| **Validation** | `104 – 116` | `13` | `10,685` | `3,612` | `1,042` | `9.75%` | `0.0874%` |
| **Test (Sealed)** | `117 – 130` | `14` | `7,286` | `2,162` | `1,197` | `16.43%` | `0.3071%` |
| **Total** | **`0 – 130`** | **131** | **58,638** | **19,156** | **7,169** | **12.23%** | **0.1218%** |

> - Tumor-positive patients: 96 in Train, 9 in Validation (the 9 define the Mark 4E validation target cohort).
> - Test holdout is strictly sealed (never used for EDA, threshold search, checkpoint selection, or loss tuning).

### 3.3 Low-Burden Split Composition (native-mask threshold 0.05%)

| Split | Zero | Low Nonzero | Combined | Share |
| :--- | :---: | :---: | :---: | :---: |
| Train | 8 | 66 | 74 | 71.2% |
| Validation | 4 | 5 | 9 | 69.2% |
| Test | 1 | 5 | 6 | 42.9% |

### 3.4 Split CSV Hashes

| Artifact | Path | SHA-256 |
| :--- | :--- | :--- |
| Train Slices CSV | `splits\train_slices.csv` | `6b0e2753dbbf780383fdf25f4421f32c496752265d83a348fbda38d2291492cc` |
| Train Volumes TXT | `splits\train_volumes.txt` | `6183e84acb5a30fd04a38875f54c6a02b4f64f588725f236ff3984ef4450a62d` |
| Validation Slices CSV | `splits\val_slices.csv` | `4b3d6abd06823239dda131add2eb407d48078fcd17b931acfbff7128425312b1` |
| Validation Volumes TXT | `splits\val_volumes.txt` | `a28fea675229371093748adb90e9c1d23dd2a5699fd6f833a902ee4681fdb9ad` |
| Test Slices CSV (Sealed) | `splits\test_slices.csv` | `6034134006793c9f5a1392aec51c3e44eb47b54d6ab2e74aca11c8166e1bca0a` |
| Test Volumes TXT (Sealed) | `splits\test_volumes.txt` | `830f8ded37afeb01347c5f222d91d253ac11cad8bc9621ce8cb04b0ef6b8d154` |

---

## 4. Slice Manifest Schema

Master manifest: `manifests\slice_manifest.csv` — **58,638 rows × 27 columns**, one row per axial slice. Every row has `verification_status=verified`, `manual_spatial_status=approved`, and unique `sample_id` (`v{NNN}_s{NNNN}`).

| Column | Type | Description |
| :--- | :--- | :--- |
| `sample_id` | String | `v000_s0000` style unique key |
| `volume_id` | Integer | Patient volume `0–130` |
| `slice_index` | Integer | Zero-indexed axial position |
| `image_path` | String | `images/v000/s0000.png` |
| `organ_mask_path` | String | `organ_masks/v000/s0000.png` |
| `tumor_mask_path` | String | `tumor_masks/v000/s0000.png` |
| `source_volume_path` / `source_segmentation_path` | String | Raw NIfTI paths |
| `source_volume_sha256` / `source_segmentation_sha256` | String | NIfTI hashes |
| `transform_applied` | String | `identity` or `rot180` |
| `image_width` / `image_height` | Integer | `256` / `256` |
| `organ_pixels` / `tumor_pixels` | Integer | Counts per slice |
| `organ_present` / `tumor_present` | Boolean | Presence flags |
| `automatic_integrity_pass` | Boolean | `True` for all rows |
| `verification_status` | String | `verified` for all rows |
| `exclusion_reason` | String | Empty (0 rows excluded) |
| `build_id` | String | `build_corrected_20260713_214847_v2` |
| `preprocessing_profile` | String | `HU[-160,240]_bilinear_image_nearest_mask_256` |
| `mask_operation_order` | String | `derive_resize_nearest_then_transform_256` |
| `split` | String | `train`, `val`, or `test` |
| `manual_spatial_status` | String | `approved` |
| `eda_nonspatial_ready` / `eda_spatial_ready` | Boolean | `True` |

---

## 5. Spatial Geometry & Physical Dimensions

Physical volume: `V = Pixel Count × Δx × Δy × Δz (mm³)`.

### 5.1 Volume Geometry Profile (117 Train + Validation Scans)

| Feature | Train (104) Median | Validation (13) Median | Cohort Range |
| :--- | :---: | :---: | :---: |
| Slices per Volume (N_z) | `263` | `816` | `[74, 987]` |
| In-Plane Spacing (Δx, Δy) | `0.772 mm` | `0.781 mm` | `[0.557, 1.000] mm` |
| Slice Thickness (Δz) | `1.000 mm` | `0.800 mm` | `[0.450, 5.000] mm` |
| FOV X/Y | `395.5 mm` | `399.9 mm` | `[285.2, 512.0] mm` |
| FOV Z | `469.5 mm` | `635.6 mm` | `[74.0, 808.0] mm` |
| Physical Liver Volume | `1,636 mL` | `1,575 mL` | `[583, 3,341] mL` |
| Physical Tumour Volume | `14.3 mL` | `3.6 mL` | `[0.0, 968.6] mL` |
| ROI Crop Area Ratio | `0.420` | `0.427` | `[0.220, 0.606]` |
| ROI Crop Height / Width | `173 px` / `156 px` | — | `[114, 235] px` / `[117, 207] px` |

### 5.2 Frozen Liver ROI Extraction Protocol (2-Stage)

1. **Liver Score Threshold**: `0.50` (Stage-1 Liver Net).
2. **Component Filtering**: `largest_3d` connected volume component.
3. **Spatial Padding**: `+16 px` in X and Y.
4. **ROI Rescaling**: to `256 × 256`.
5. **Coverage Verification**: 100% tumor-pixel containment (Train + Validation, `152,763 / 152,763` px in V116).

---

## 6. Radiometrics & HU Windowing

### 6.1 Preprocessing Profile: `HU[-160,240]_bilinear_image_nearest_mask_256`

```
Raw NIfTI (.nii) ──► Clip HU [-160, +240] ──► Min-Max Scale [0,1] ──► Bilinear Resize to 256×256 PNG
```

- Broad Abdominal Window: `[-160, +240] HU` (8-bit grayscale PNG, 256×256)
- Image Interpolation: bilinear; Mask Interpolation: nearest-neighbor
- Soft Tissue/Liver Window (analytical only): `[0, +200] HU`

### 6.2 Normalized Intensity Statistics (8-bit [0,255])

| Region | Mean | Std | Notes |
| :--- | :---: | :---: | :--- |
| All Image Pixels | `44.5` | `84.1` | Full axial distribution |
| Background | `44.4` | `84.1` | Median HU `-887` (train) / `-908` (val) |
| Tumor Tissue | `109.1` | `102.1` | Hyper-intense post-windowing |
| Median Tumor-minus-Liver Contrast | — | — | `-34 HU` (train) / `-44 HU` (val) |
| Robust CNR | — | — | `-1.518` (train) / `-1.881` (val) |

### 6.3 HU Appearance Profile (Train vs Validation)

| Metric | Train Median | Val Median | Std. Shift |
| :--- | :---: | :---: | :---: |
| Tumour HU Median | `+67 HU` | `+59 HU` | `-0.16σ` |
| Tumour HU IQR | `36 HU` | `39 HU` | `-0.04σ` |
| Liver HU Median | `+99 HU` | `+106 HU` | `+0.36σ` |
| Tumour-minus-Liver Contrast | `-34 HU` | `-44 HU` | `-0.55σ` |
| Robust CNR | `-1.52` | `-1.88` | `-0.46σ` |
| Background HU Median | `-887 HU` | `-908 HU` | `-0.43σ` |

---

## 7. 3D Lesion Morphology & Stratification Bins

**845 distinct 3D 6-connected lesion components** across the cohort (637 train, 208 validation).

| Lesion Metric | Value |
| :--- | :---: |
| Train 3D Lesions | `637` |
| Validation 3D Lesions | `208` |
| 3D Lesion Volume (median) | `0.383 mL` (range `[0.00035, 968.60] mL`) |
| Equivalent Spherical Diameter (median) | `9.01 mm` (range `[0.87, 122.70] mm`) |
| Axial Span (median) | `7 slices` |
| Lesions per Patient (median) | `4` (max `70`) |
| Max Tumour-to-Liver Ratio | `0.314` |

### 7.1 Train-Derived Lesion Stratification Bins (`train_derived_lesion_bins.json`)

Fit strictly on 637 train lesions — no validation leakage.

| Stratum | Volume Range | Spherical Diameter | Notes |
| :--- | :---: | :---: | :--- |
| **Q1 (Very Small/Small)** | `(−∞, 0.173 mL]` | `(−∞, 6.92 mm]` | High-resolution feature retention needed |
| **Q2 (Medium-Small)** | `(0.173, 0.673 mL]` | `(6.92, 10.87 mm]` | Subtle parenchymal lesions |
| **Q3 (Medium-Large)** | `(0.673, 3.944 mL]` | `(10.87, 19.60 mm]` | Moderately defined focal lesions |
| **Q4 (Large/Massive)** | `(3.944, +∞ mL)` | `(19.60, +∞ mm)` | Extensive deformation (up to 266.35 mL) |

### 7.2 Slice-Level Reference Groups (reporting; do not interchange with qcut quartiles)

| Group | Tumor Pixels per Slice |
| :--- | :---: |
| Tiny | `1–51` |
| Small | `52–196` |
| Medium | `197–699` |
| Large | `≥700` |

> V104 = multifocal, very-low-contrast phenotype; V116 = large solitary, weak-contrast phenotype. Both have train analogs; neither failure is explained by ROI clipping.

---

## 8. Data Forensics: Fixes Applied

| Anomaly | Root Cause | Verified Fix |
| :--- | :--- | :--- |
| **47 Volume Orientation Flips** | Image/mask 180° in-plane mismatch in Kaggle/HF imports | `rot180` transform on Volumes `83–99`, `101–130`; identity on `0–82`, `100` |
| **Slice Offset Discrepancy** | Suspected z-offset | Offset = `0`; problem was orientation, not offset |
| **4× Tumor Burden Error** | `512×512` denominator (`262,144 px`) on `256×256` masks (`65,536 px`) | Native-mask denominator; train burden `0.0251% → 0.1003%` |
| **30,105 Mask Values >255** | Multi-instance label overflow | `instance mod 256` matching all 58,638 rows |
| **Tumor-Positive Slice Count Mismatch** | Legacy CSV said 7,114 | Corrected manifest records **`7,169`** |

### Spatial Rotation Matrix (Rot180)

```
[x']   [-1  0 255] [x]
[y'] = [ 0 -1 255] [y]
[1 ]   [ 0  0   1] [1]
```

---

## 9. Validation & Metric Contract

### 9.1 Required Metrics (aggregate Dice alone is insufficient)

- Global pixel micro-Dice
- Mean / median / worst positive-patient micro-Dice
- Per-patient Dice (esp. Volumes 104 and 116)
- Pixel precision & recall
- Positive-slice recall
- Positive slices predicted completely empty (%)
- Empty-slice false-positive rate (%)
- Lesion-size-stratified detection & Dice (Q1–Q4)
- Threshold sensitivity
- Expected-vs-actual targets

### 9.2 Core Formulas

| Metric | Formula |
| :--- | :--- |
| Hard Dice | `(2 × intersection + ε) / (predicted + true + ε)` |
| Positive-slice recall | `slices with any TP overlap / all tumor-positive slices` |
| Positive predicted-empty % | `tumor-positive slices with 0 predicted px / all tumor-positive × 100` |
| Empty-slice FP % | `tumor-negative slices with any predicted px / all tumor-negative × 100` |

### 9.3 Thresholds

- Default hard tumor threshold: `0.50`
- Standard diagnostic sweep: `0.30–0.90` in `0.05` steps
- Next calibration must extend lower: `0.10–0.60` (multi-task model over-suppressed at 0.50)
- **Frozen final policy threshold: `0.70`** (pixelwise maximum fusion)

### 9.4 Multi-Task Epoch-10 Targets (validation)

| Metric | Direction | Target |
| :--- | :--- | :---: |
| Mean positive-patient Dice | Higher | `0.406915` |
| Volume 104 Dice | Higher | `0.50` |
| Volume 116 Dice | Higher | `0.05` |
| Q1 smallest-lesion detection | Higher | `45%` |
| Positive predicted-empty rate | Lower | `20%` |
| Empty-slice false-positive rate | Lower | `15%` |
| Mean liver Dice | Higher | `0.90` |

### 9.5 Overfit Gate (passed)

- 16 selected slices; max 150 epochs; completed 76 epochs
- Required hard micro-Dice `0.80`; achieved **`0.816306`** @ threshold 0.50

### 9.6 Test-Lock Contract

- Test patients disjoint from train/val; test images never accessed in saved gate results
- Test run allowed **only after all validation targets pass**; policy frozen before opening
- Test results must never be used to choose thresholds or modify the model

---

## 10. Model Benchmarks: Mark 1 → Mark 4E

### 10.1 Progression

| Phase | Notebook | Core Finding / Milestone |
| :--- | :--- | :--- |
| **Mark 1** | `mark_1_probability_contrast_localization_diagnostic.ipynb` | Identified hypodense lesion suppression on small components |
| **Mark 2** | `mark_2_roi_multiwindow_feasibility.ipynb` | Validated Stage-1 liver crop bounding box (reduction ratio 0.42) |
| **Mark 3** | `mark_3_two_stage_multiwindow_overfit.ipynb` | Hard micro-Dice **`0.900557`** (17 epochs, 100% containment) |
| **Mark 4/4B** | `mark_4_two_stage_validation_smoke.ipynb` | Revealed high positive predicted-empty rate (`36.85%`) |
| **Mark 4C/4D** | `mark_4c_two_channel_recall_ablation.ipynb`, `mark_4d_metric_reconciliation_v116_diagnostic.ipynb` | Isolated V116 hypodense lesion localization failure |
| **Mark 4E** | `mark_4e_checkpoint_fusion_validation.ipynb` | **PASSED ALL 6 VALIDATION TARGETS** @ threshold 0.70 |

### 10.2 Checkpoint Fusion Policy

$$P_{\text{fused}}(x, y) = \max\left(P_{\text{control}}(x, y), P_{\text{recall\_loss}}(x, y)\right)$$

- Control checkpoint: `mark 1/mark_4_outputs/mark_4_best.pth`
- Recall checkpoint: `mark 1/mark_4c_outputs/recall_loss_best.pth`
- Threshold: `0.70`; post-processing: none

### 10.3 Mark 4E Controlling Validation Gate Results (9 tumor-positive validation patients)

| Metric | Target | Mark 4E Result | Status |
| :--- | :---: | :---: | :---: |
| Mean Positive-Patient Dice | ≥ `0.3329` | **`0.377087`** | **PASS** |
| V104 Patient Dice | ≥ `0.0500` | **`0.116627`** | **PASS** |
| V116 Patient Dice | ≥ `0.0100` | **`0.010474`** | **PASS** |
| Q1 Small Lesion Detection Rate | ≥ `35.0%` | **`50.57%`** | **PASS** |
| Positive Predicted-Empty Rate | ≤ `35.0%` | **`27.45%`** | **PASS** |
| Empty-Slice False Positive Rate | ≤ `20.0%` | **`5.55%`** | **PASS** |

### 10.4 Step 02 Fusion Freeze Confirmation (2026-08-03, `VALIDATION_FREEZE_PASS`)

- Mean Dice (9 positive patients): `0.37706060` (bootstrap 95% CI `0.19792987–0.55420929`)
- V104: `0.11647368`; V116: `0.01047342`
- Q1 detection: `50.570342%`; positive predicted-empty: `27.447217%`; empty-slice FP: `5.548066%`
- Determinism max diff: `0.0`; fresh/historical mean score error ≈ `1.09e-7`; **maximum reproduced metric difference from Mark 4E: `0.00015329`** (explains the small delta vs §10.3 — expected fresh-inference reproduction error)
- **Known limits**: V116 margin only ~0.000473 above floor; smallest component-quartile detection 13.04%; Q2 76.09%, Q3 91.30%, Q4 100%

### 10.5 Intermediate Milestone Results (Mark 4B → 4C → 4D — from `MARK_1_PROGRESS_TRACKER.md`)

**Progress tracker statuses**: Mark 4B `~70%` → Mark 4D `~73%` → Mark 4E `~77%` workflow complete (as of 2026-08-03).

#### Mark 4B — ROI probability diagnostics (threshold sweep 0.05–0.70)

| Metric | @ threshold 0.60 | Temporary target | Status |
| :--- | :---: | :---: | :---: |
| Mean patient Dice | `0.3646` | ≥ `0.3329` | **Pass** |
| Volume 104 Dice | `0.0646` | ≥ `0.0500` | **Pass** |
| Volume 116 Dice | `0.0104` | ≥ `0.0100` | **Pass** |
| Q1 lesion detection | `42.59%` | ≥ `35.00%` | **Pass** |
| Positive slices predicted empty | `37.04%` | ≤ `35.00%` | **Fail** |
| Empty-slice FP rate | `3.40%` | ≤ `20.00%` | **Pass** |

> Threshold-only tuning could not fix recall: positive predicted-empty stayed between `36.28–37.04%` across the whole sweep. Bootstrap 95% CI for mean patient Dice at 0.60 ≈ `0.189–0.531` (9-patient validation).

#### Mark 4C — Bounded ablation (control vs two-channel vs recall-loss, 5 epochs)

| Arm | Best Epoch | Mean Patient Dice | V104 | V116 | Q1 Detection | Pos-Empty | Empty FP | Targets Passed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Control | 5 | `0.3639` | `0.0672` | `0.0108` | `42.59%` | `36.85%` | `3.42%` | 5/6 |
| Two-channel | 1 | `0.0837` | ≈0 | `0.0843` | `0.00%` | `60.46%` | `5.82%` | 2/6 |
| Recall-loss | 2 | `0.2603` | `0.0091` | `0.0019` | `39.54%` | `35.70%` | `3.44%` | 2/6 |

- Two-channel arm collapsed to all-background on tumor-positive patients → **rejected**.
- Recall-loss improved positive-empty (`36.85% → 30.13%`), Q1 (`42.59% → 47.91%`), V104 (`0.0672 → 0.1038`) but **failed V116** (Dice `0.00085`).
- *Metric-correction note*: initial experimental-arm `mean_patient_dice` included 4 tumor-empty val patients; recomputed over the same 9 positive patients → recall-loss mean `0.3749`, two-channel ≈ 0.

#### Mark 4D — Metric reconciliation & V116 localization diagnostic

- No checkpoint/threshold pair passed all 6 continuation targets.
- Recall-loss @ 0.60 passed 5 targets: mean Dice `0.3766`, V104 `0.1002`, Q1 `47.91%`, pos-empty `30.52%`, empty-FP `4.79%`; **V116 failed** at `0.00071`.
- Control kept V116 above target through threshold 0.65 but failed positive predicted-empty at every threshold.
- **V116 failure is NOT ROI clipping** (100% of 152,763 tumor pixels inside frozen ROI) and NOT small-lesion-limited (median truth-region probability = 0 in every V116 lesion quartile; only 3.0% of V116 positive slices detected by recall-loss @ 0.50).

#### Mark 4E — Checkpoint-fusion selection (7 fusion policies × 14 thresholds)

- Pixelwise maximum fusion @ 0.70 passed **all 6** temporary targets (see §10.3).
- Mean, 75/25 and 25/75 weighted fusion also passed; max fusion selected by predeclared highest-mean-Dice rule.
- V116 margin remains narrow → temporary validation pass, **not** robust-generalization evidence.

---

## 11. Complete Experiment Results (Practice Ablations)

All values from saved JSON/CSV artifacts under `Practice/*_outputs`. `≈0` = effectively zero.

| Experiment | Best Epoch | Global Dice | Mean Patient Dice | V104 Dice | V116 Dice | Q1 Detection | Pos-Empty | Empty FP | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 16-slice overfit | 76 | `0.816306` | — | — | — | — | — | — | Pass |
| 5-epoch smoke | 5 | `0.461613` | — | — | — | — | `27.74%` | `14.42%` | Pass |
| Patient-aware baseline | 9 | `0.558357` | `0.383069` | `0.197613` | `≈0` | — | — | — | Revise |
| Lesion-balanced sampler | 7 | `0.524200` | `0.356189` | `0.152444` | `0.001856` | `53.99%` | `19.29%` | `17.83%` | Fail |
| Pure Focal-Tversky | 1 | `≈0` | `≈0` | `≈0` | `≈0` | `0%` | `100%` | `0%` | Collapsed |
| Composite loss (0.75 FD + 0.25 FT) | 5 | `0.568154` | `0.322939` | `0.005005` | `0.051481` | `29.66%` | `41.46%` | `3.03%` | Fail |
| Organ-assisted intensity | 8 | `0.546132` | `0.406915` | `0.586500` | `0.026782` | `60.08%` | `8.25%` | `47.42%` | Fail (not deployable) |
| Adjacent-slice 2.5D | 10 | `0.553527` | `0.355938` | `0.000636` | `0.000932` | `25.48%` | `42.99%` | `2.37%` | Fail |
| 3D post-processing | val-only | — | `0.406964` | `0.586555` | `0.026794` | `60.08%` | `8.25%` | `47.43%` | Fail |
| Multi-task liver/tumor | 8 | — | `0.332854` | `≈0` | `≈0` | `28.14%` | `46.16%` | `3.25%` | Fail (1/7 targets) |

### Key Ablation Conclusions

1. **Organ-assisted normalization** produced the strongest mean patient Dice (0.4069) and V104 (0.5865), but **leaks ground-truth organ info at inference** → diagnostic only, not deployable.
2. **2.5D context & multi-task learning** suppress false positives but also suppress true tumors.
3. **V104** is highly preprocessing/domain sensitive (appearance shift score `6.6768` > train P95 `6.4022`).
4. **V116** (shift score `3.7398`) is persistently difficult, not explained by the same appearance-shift signal.
5. **Multi-task liver gating was a no-op**: raw and liver-gated predictions were numerically identical (gate removed 0 pixels) — remaining errors are intra-liver discrimination errors.

---

## 12. Multi-Task Model: Epoch Trajectory & Patient Detail

### 12.1 Epoch Trajectory (best = epoch 8)

| Epoch | Train Loss | Val Loss | Liver Dice | Mean Patient Dice | V104 | V116 | Pos-Empty | Empty FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `0.3895` | `0.4472` | `0.8461` | `0.2960` | `0.0159` | `0.0027` | `20.63%` | `15.64%` |
| 2 | `0.2065` | `0.4387` | `0.8669` | `0.3028` | `≈0` | `≈0` | `46.07%` | `3.40%` |
| 3 | `0.1287` | `0.4406` | `0.8801` | `0.2959` | `≈0` | `0.0026` | `49.42%` | `2.78%` |
| 4 | `0.1243` | `0.4352` | `0.8848` | `0.3138` | `≈0` | `≈0` | `50.00%` | `3.00%` |
| 5 | `0.1212` | `0.4352` | `0.8884` | `0.3133` | `≈0` | `0.0004` | `47.50%` | `3.09%` |
| 6 | `0.1182` | `0.4178` | `0.8785` | `0.3039` | `0.0286` | `≈0` | `51.15%` | `2.59%` |
| 7 | `0.1160` | `0.4236` | `0.8902` | `0.3180` | `0.00003` | `≈0` | `49.04%` | `2.24%` |
| **8** | `0.1126` | `0.4099` | `0.8863` | **`0.3329`** | `≈0` | `≈0` | `46.16%` | `3.25%` |
| 9 | `0.1120` | `0.4103` | `0.8796` | `0.3296` | `0.00014` | `0.00014` | `42.42%` | `3.94%` |
| 10 | `0.1102` | `0.4120` | `0.8881` | `0.3296` | `≈0` | `0.00012` | `45.87%` | `2.89%` |

### 12.2 Best Multi-Task Patient Results (epoch 8)

| Volume | Tumor Pixels | Predicted Pixels | Dice | Pos-Slice Recall | Pos-Empty | Empty FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 104 | 69,032 | 307 | `≈0` | `0%` | `97.54%` | `0.76%` |
| 116 | 152,763 | 534 | `≈0` | `0%` | `96.24%` | `1.42%` |
| 112 | 295 | 4,107 | `0.0672` | `38.46%` | `34.62%` | `4.97%` |
| 107 | 1,473 | 4,072 | `0.0775` | `21.13%` | `77.46%` | `8.71%` |
| 111 | 1,830 | 3,468 | `0.3103` | `40.22%` | `56.52%` | `3.59%` |
| 113 | 21,076 | 24,092 | `0.5811` | `74.81%` | `20.00%` | `2.28%` |
| 108 | 363,698 | 224,790 | `0.6005` | `71.29%` | `22.28%` | `0.46%` |
| 109 | 18,200 | 19,407 | `0.6573` | `77.10%` | `16.03%` | `3.36%` |
| 110 | 31,120 | 22,911 | `0.7017` | `80.77%` | `19.23%` | `0.15%` |

> Volumes 105, 106, 114, 115 have no tumor ground truth. At best epoch, volume 114 had no predicted tumor.

### 12.3 Best Multi-Task Size-Quartile Results

| Quartile | Slices | Mean Dice | Median Dice | Detection | Predicted Empty |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Q1 smallest | 263 | `0.1556` | `≈0` | `28.14%` | `66.16%` |
| Q2 | 258 | `0.3273` | `0.3441` | `58.14%` | `36.43%` |
| Q3 | 260 | `0.4274` | `0.6255` | `61.15%` | `37.31%` |
| Q4 largest | 261 | `0.3250` | `≈0` | `49.81%` | `44.44%` |

### 12.4 Appearance-Shift Diagnostic

| Measure | Value |
| :--- | :---: |
| Sampled slices | `3,106` |
| Training P95 shift threshold | `6.4022` |
| Volume 104 shift score | `6.6768` (outlier) |
| Volume 116 shift score | `3.7398` (not outlier) |

---

## 13. One-Time Locked Test Evaluation (Step 04)

**Run UUID**: `871d289b-bf6b-4346-978f-2df02ade26ab` · Authorized `2026-08-05T11:39:37.3247843Z` · Ledger sealed, rerun prohibited.

| Metric | Value |
| :--- | :---: |
| Result Level | `FINAL_TEST_COMPLETE` |
| Formal Minimum Acceptance | **`False`** (failed) |
| Global Dice | `0.767696` |
| Mean Positive-Patient Dice | `0.507297` (bootstrap 95% CI `0.344175–0.659411`) |
| Median Positive-Patient Dice | `0.5651` |
| Minimum Positive-Patient Dice | `0.000000` (V121) |
| Global Pixel Precision | `0.844394` |
| Global Pixel Recall | `0.703771` |
| Train-edge Q1 Detection (slice) | `47.552%` |
| Lesion-volume Q1 Detection | `60.29%` (mean matched Dice `0.238`); Q2 `98.00%`, Q3 `97.37%`, Q4 `100.00%` |
| Positive Predicted-Empty | `9.942%` |
| Empty-Slice False Positives | `3.810%` |
| Formal Acceptance Failure | **V121** — 526 tumor pixels / 3 connected lesions, 100% ROI containment, no truth-region score ≥ 0.70 (recognition failure, not ROI clipping) |
| Empty-Tumor Patient | V119 generated 968 FP pixels (excluded from positive mean, included in empty-slice FP) |

> **Decision**: `REPORT_FINAL_TEST_GATE_FAILURE_NO_TUNING_NO_RERUN`. Final report-only evidence; no threshold/checkpoint/fusion/post-processing/training change permitted.

---

## 14. External Evaluation: 3D-IRCADb-01 (Steps 12–21)

### 14.1 Dataset Facts

| Attribute | Value |
| :--- | :--- |
| Source | IRCAD (Research Institute against Digestive Cancer), France |
| License | CC BY-NC-ND 4.0 International (official page, 20 patients: 10 women / 10 men) |
| Citation | Soler L, et al. "3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database." IRCAD Technical Report (2010) — `Soler2010IRCADb` |
| Volume Count | **20** anonymized abdominal CT scans (2,827 total axial slices; 2,823 evaluated) |
| Tumor-Positive / Negative | 15 positive / 5 negative controls, zero exclusions |
| Archive | 20 official patient ZIP archives, `820,269,486` bytes; archive-set SHA-256 `55cd6722da6c02f8365ab566603db116128e0d854782d2a174b85056f4a5ba96` |
| Normalized Geometry | 74–260 slices; 0.561–0.873 mm in-plane; 1–4 mm slice spacing; tumor-in-liver containment ≥ 98.545% |
| Parity Pipeline | DICOM → NIfTI → HU `[-160,240]` → 256×256 (identical to LiTS preprocessing) |

### 14.2 External Evaluation Result (Step 16, run UUID `61d1f140-c8b4-436a-928a-1e4b6f7c0b56`)

**Gate: PASS (11/11 mandatory rows)** · Assessment: `EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS`

| Metric | Value |
| :--- | :---: |
| Global Dice | `0.847977` |
| Pixel Precision / Recall | `0.823699` / `0.873729` |
| Mean Positive-Patient Dice | `0.711196` (bootstrap 95% CI `0.567803–0.827823`) |
| Median Positive-Patient Dice | `0.835514` |
| Minimum Positive-Patient Dice | `0.012992` (`ircadb_18`) |
| Q1 Positive-Slice Detection | `50.00%` |
| Positive Predicted-Empty Rate | `9.86%` |
| Empty-Slice False-Positive Rate | `11.57%` |
| Smallest-Lesion Detection | `63.64%` (14/22 components) |
| Burden Stratum Means | Q1 `0.717752`; Q2 `0.544269` (min 0.012992); Q4 `0.877186` |

### 14.3 Required Caveats

- Worst positive case `ircadb_18`: Dice `0.012992`, recall `1.41%` (genuine accepted-truth failure under frozen contract).
- All 5 tumor-negative controls had ≥1 predicted pixel; total standardized FP volume `52.936 mL` (cases 7 & 14 dominate).
- **Case 7**: 86.39% of predicted pixels overlap excluded generic `tumor` mask. **Case 14**: 88.93% overlap `metastasectomie` (source-label Dice `0.851925`). **Case 20**: 71.66% overlap gallbladder mask. (Step 18 source-label concordance — annotation concordance, not biological truth.)
- Tumor-negative labels exclude adrenal/generic non-hepatic tumor folders; no biological reinterpretation without expert/source review.
- Reliability curve & ECE are background-dominated descriptive diagnostics, not clinical calibration claims.
- Cross-cohort "outperformed" comparisons are prohibited as inferential claims (different acquisition, prevalence, denominators, label semantics).

---

## 15. Part 2 Research Pipeline Status (Steps 00–21)

| Step | Phase | Result Level | Status |
| :--- | :--- | :--- | :--- |
| Step 00 | Model Improvement Program | Multiple (see §16) | Phase 0 complete; Phase 1 smoke failed gate |
| Step 01 | Pretraining Dataset Characterization | `DIAGNOSTIC_COMPLETE` | 12/12 mandatory targets; `PASS_FREEZE_DATA_CARD...` |
| Step 02 | Fusion Freeze Confirmation | `VALIDATION_FREEZE_PASS` | 6/6 targets; deterministic inference verified |
| Step 03 | Final Inference Policy Freeze | `VALIDATION_FREEZE_PASS` | 10/10 requirements; 38 artifacts checksummed |
| Step 04 | One-Time Locked Test Evaluation | `FINAL_TEST_COMPLETE` | Formal acceptance **failed** (V121); no rerun |
| Step 05 | Final Research Package | `FINAL_PROJECT_COMPLETE` | 12/12 requirements; 23 outputs |
| Step 06 | Manuscript Submission Readiness | `DIAGNOSTIC_COMPLETE` | Submission readiness **false** (11 manual items) |
| Step 07 | Evidence & Declaration Scaffold | `MANUSCRIPT_EVIDENCE_SCAFFOLD_COMPLETE` | 10/10 scaffold + 10/10 sealed-input checks |
| Step 08 | Venue Selection & Owner Intake | `VENUE_DECISION_SUPPORT_COMPLETE` | Preliminary rank: BMC Medical Imaging 4.05/5; no venue selected |
| Step 09 | Internal Peer Review | `INTERNAL_PEER_REVIEW_COMPLETE` | `SHARE_WITH_CAVEATS`; 11/11 phase checks |
| Step 10 | Related Work Evidence | `RELATED_WORK_EVIDENCE_COMPLETE` | 13 primary sources; quantitative rankings prohibited |
| Step 11 | Owner Submission Gate | `OWNER_INPUT_REQUIRED` | Owner fields 0/14 (inactive by owner choice) |
| Step 12 | Public External Data Audit | `PUBLIC_EXTERNAL_DATA_AUDIT_COMPLETE` | 3D-IRCADb-01 recommended (4.25/5) |
| Step 13 | 3D-IRCADb Ingestion & QC | `EXTERNAL_SOURCE_INGESTION_QC_PASS` | 20/20 patients; archive hash verified |
| Step 14 | Normalized Conversion & Parity QC | `EXTERNAL_NORMALIZED_CONVERSION_QC_PASS` | 12/12 checks; 15/20 tumor cases reconciled |
| Step 15 | Frozen External Evaluation Contract | `EXTERNAL_EVALUATION_CONTRACT_FROZEN_AWAITING_AUTHORIZATION` | 10/10 readiness |
| Step 16 | One-Time External Evaluation | `EXTERNAL_EVALUATION_COMPLETE` | **PASS** 11/11; UUID `61d1f140...`; sealed |
| Step 17 | External Evidence Validation & Data Card | `EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS` | Headline metrics independently recomputed |
| Step 18 | Source-Label Concordance & Adjudication | `SOURCE_LABEL_CONCORDANCE_COMPLETE_EXPERT_REVIEW_REQUIRED` | 72 DICOM mask folders aligned; expert review pending |
| Step 19 | External Evidence Manuscript Integration | `MANUSCRIPT_EXTERNAL_EVIDENCE_UPDATED_OWNER_INPUT_REQUIRED` | Cross-cohort comparisons descriptive only |
| Step 20 | External Citation Verification | `EXTERNAL_CITATION_VERIFIED_OWNER_DECLARATIONS_REQUIRED` | `Soler2010IRCADb` verified; no DOI fabricated |
| Step 21 | Terminal Evidence Chain & Handoff | `PROJECT_EVIDENCE_CHAIN_VERIFIED_OWNER_ACTIONS_PENDING` | 20/20 phases; 192 artifact mappings verified |

### Step Signatures (combined SHA-256)

| Step | Signature |
| :--- | :--- |
| Step 14 | `a023b54f4b93ca935162824b6e51d3e70a405383ae38ba5b89c675a24839dc90` |
| Step 15 | `aaca93be9d1d64e63807fb10f642b98cd0371504667f55ad69a14d8f1246f77d` |
| Step 16 | `4fb3d6d14010a982912c9ca8e85854e326960fa4a60543da47a378c6fd1b13aa` |
| Step 17 | `4b0f2b13eef3f5a70a41eca94b826703f3daeac85b17b354886c3357ac4c3b48` |
| Step 18 | `1f1f222d300e689a5e36787b4b7da60b0b27ddb594bab8d8aad736fe3ed35b6f` |
| Step 19 | `d6075d9fc6d5d802316f2e3e7d9f7ee3f379ae9fc6919053e292fb496b3e0fb3` |
| Step 20 | `316c298dcf6e4f027de9d039ce7c2aa8637bc6e37971613bb1b46819008a751d` |
| Step 21 | `980669a07052035505b4239fe6a7c5e78958c9eb6fbf15381add78f924688282` |

### Frozen Inference Policy (Step 03, immutable)

- Input: one broad `[-160,240]` HU channel, uint8 / 255
- ROI: predicted-liver threshold 0.50 → largest 3D component → padding 16 → 256×256 (full-image fallback for empty ROI)
- Fusion: pixelwise maximum of control + recall-loss probabilities
- Global threshold: `0.70`; post-processing: none; ε = 1e-6; seed 42
- Q1 test-slice definition: 1–51 tumor pixels (51 = train-only 25th percentile)

---

## 16. Model Improvement Program (Step 00)

New development-only research programme (Phases 0–5) to improve tumor segmentation on the local corrected LiTS build, independent of the completed Mark 4E baseline.

### Phase 0 — Contract & Holdout Seal (COMPLETE)

- **EVAL_VOLUMES (8-volume sealed holdout)**: `[4, 25, 44, 64, 83, 84, 90, 100]`
  - Large-lesion stratum: `[4, 100]`; Low-burden: `[83, 25]`; V104 analogs: `[64, 90]`; V116 analogs: `[84, 44]`
- Dev train volumes: **96** (of 104 train); validation: 13 (unchanged); test: locked
- Status: `STEP_0_CONTRACT_AND_HOLDOUT_SEALED`; decision `PROCEED_TO_STEP_1_SMOKE_GATE`
- Sampling: patient-aware uniform (1/volume_count) × positive-slice multiplier 3.0; C2 analog boost cap 4.0 (top-5 training analogs of V104/V116)

### Phase 1 — Smoke Gate (EXECUTED, **FAILED**)

16-slice overfit canary (target hard micro-Dice ≥ 0.80 for ALL arms) — **`HALT_REVIEW_ARM`**:

| Arm | Loss | Epochs | Hard Micro-Dice | Result |
| :--- | :--- | :---: | :---: | :--- |
| Control | FocalDiceLoss | 60 | `0.0706` | Below 0.80 gate |
| C1 high-res ROI | FocalDiceLoss | 60 | `0.1544` | Below 0.80 gate |
| C2 analog sampler | FocalDiceLoss | 60 | `0.0915` | Below 0.80 gate |
| C3 capped-recall | StabilityBoundedRecallLoss | 60 | `0.0615` | Below 0.80 gate |

> All four candidates failed to overfit the 16-slice deterministic subset. This is a trainability canary failure — the program halted for arm review before Step 2 training.

### Roadmap Decoder (Phase 0 contract)

- Step 2 train: batch 4 + grad accumulation, AdamW, cosine, LR 3e-4 (C3: 1e-4), epochs 5–10
- Step 3 select: per-candidate validation on 9-positive-patient set; selector passes each of the 6 temporary targets
- Step 4 holdout: single frozen run on EVAL_VOLUMES; sealed one-time ledger
- Step 5 report

---

## 17. Configurations & Framework Parameters

### 17.1 Shared Reproducibility Settings

- Random seed: `42`; Framework: PyTorch; Mixed precision: CUDA where supported
- Deterministic cuDNN: enabled; benchmark mode: disabled
- Gradient clipping: max norm `5.0` (stabilized continuations)
- Validation augmentation: none; Test loader: locked
- Primary architecture: **MobileNetV2-U-Net**; pretrained encoder: `False` (recorded controlled gates)

### 17.2 Training Parameters by Main Stage

| Stage | Epoch Plan | Batch | Val Batch | LR | Min LR | Pos Weight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 16-slice overfit | max 150 | 4 | — | `1e-3` | — | deterministic subset |
| 5-epoch smoke | 5 | 8 | 16 | `3e-4` | — | 4 |
| Patient-aware baseline | 25 plan; decision @10 | 8 | 16 | `3e-4` | `3e-6` | 3 |
| Intensity robustness | 25 plan; decision @10 | 8 | 16 | `3e-4` | `3e-6` | 3 |
| Multi-task | 15 plan; decision @10 | 8 | 16 | `1e-4` | `2e-6` | 3 |

Schedulers: cosine annealing; Optimizer: AdamW.

### 17.3 Losses Evaluated

| Loss | Configuration | Outcome |
| :--- | :--- | :--- |
| **Focal-Dice** | alpha `0.75`, gamma `2.0`, weight `0.5`/`0.5` | Strongest stable general baseline |
| **Focal Tversky** | alpha `0.40`, beta `0.60`, gamma `0.75` | Collapsed to empty predictions |
| **Stabilized Composite** | `0.75` Focal-Dice + `0.25` Focal-Tversky | No collapse; mean patient Dice fell |
| **Multi-task loss** | tumor `0.72`, liver `0.23` (BCE+soft Dice), containment `0.05` | Stable; tumor over-suppressed |

### 17.4 Augmentation

- Paired geometric: horizontal flip p `0.30`; affine p `0.60` (rotation ±10°, translation 5%, scale 0.95–1.05)
- Intensity robustness: gamma `0.85–1.15` p `0.50`; Gaussian noise σ `0–0.025` p `0.35`

### 17.5 Thermal & Interruption Controls

- Desired start < `84°C`; emergency limit `90°C`; cooldown polling every 30 s; max bounded cooldown 30 min
- Checkpoint after every epoch; exact RNG resume (detached CPU uint8)
- Observed: epoch-end GPU temps `86–87°C`; next-start cools to `73–77°C`

### 17.6 Config Files (project YAML)

| File | Purpose | Key Values |
| :--- | :--- | :--- |
| `configs/datasets/lits_verified_eda.yaml` | Canonical LiTS dataset config | root = build_corrected_20260713_214847_v2; manifest SHA-256 `575a6fc...`; counts 131/58,638/19,156/7,169; 84 identity / 47 rot180 |
| `configs/datasets/ircad.yaml` | 3D-IRCADb stub | `status: not_implemented` (superseded by Steps 13–16 ingestion) |
| `configs/baseline.yaml` | Baseline training | MobileNetV2UNet 1ch→1ch; batch 8; LR 0.001; 50 epochs; HU window `[-100,400]`; CLAHE clip 2.0 grid 8×8 |
| `configs/upre_net.yaml` | UP³RE-Net research | ensemble_size 3; 2-stage bagging + UWACL; stage1 25 epochs pos_weight 10; stage2 beta 5.0 tau 0.1 |

> ⚠️ Note: `configs/baseline.yaml` HU window `[-100,400]` and `results/` report stats (7,114 tumor-positive slices, imbalance 1140.39:1, all-pixel mean intensity 44.86 ± 84.22) come from an **earlier pipeline**; the **corrected canonical** values are 7,169 positive slices, 822:1, HU `[-160,240]`, mean intensity 44.5 ± 84.1. The corrected manifest/build takes precedence.
>
> **Imbalance-ratio reconciliation** (three values appear because they come from three different pipelines, not a typo): `822:1` = corrected canonical build (§1, §3.1); `~937:1` = MedSegX/UP³RE research track (§22.1); `1140.39:1` = earlier framework `results/` reports (§17.7). Use the `822:1` canonical value for official claims.

### 17.7 Additional Framework Results (`results/`)

Supplementary results from the earlier framework pipeline (see **[§23.3](#23-medsegx-research-changelog--sprint-results--ablations)** for the full table and statistical tests): UP³RE 3-member ensemble Dice `0.8867 ± 0.3091` vs single MobileNetV2-UNet `0.8787 ± 0.3265`. **These are from the earlier pipeline — the controlled gates in Steps 01–21 use the corrected build.**

---

## 18. Cryptographic Hashes & Verification Fingerprints

| Artifact | Path | SHA-256 |
| :--- | :--- | :--- |
| Master Slice Manifest | `manifests\slice_manifest.csv` | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |
| Train Slices CSV | `splits\train_slices.csv` | `6b0e2753dbbf780383fdf25f4421f32c496752265d83a348fbda38d2291492cc` |
| Train Volumes TXT | `splits\train_volumes.txt` | `6183e84acb5a30fd04a38875f54c6a02b4f64f588725f236ff3984ef4450a62d` |
| Validation Slices CSV | `splits\val_slices.csv` | `4b3d6abd06823239dda131add2eb407d48078fcd17b931acfbff7128425312b1` |
| Validation Volumes TXT | `splits\val_volumes.txt` | `a28fea675229371093748adb90e9c1d23dd2a5699fd6f833a902ee4681fdb9ad` |
| Test Slices CSV (Sealed) | `splits\test_slices.csv` | `6034134006793c9f5a1392aec51c3e44eb47b54d6ab2e74aca11c8166e1bca0a` |
| Test Volumes TXT (Sealed) | `splits\test_volumes.txt` | `830f8ded37afeb01347c5f222d91d253ac11cad8bc9621ce8cb04b0ef6b8d154` |
| Corrected `dataset_version.json` | build root | `fb39183db4d59b8010dbb77426e4ba2166a1316c6b2da6925996a41739fd22bf` |
| Corrected `dataset_readiness.json` | build root | `2dad7f66e37edc57be4e27f1e68707f2d48a5bd432f962ba703734db7bf92bc8` |
| Source Registry JSON | `00_source_registry\source_registry.json` | `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173` |
| **Control Checkpoint** | `mark 1/mark_4_outputs/mark_4_best.pth` | `9b0c7749af66b0fc3808f6757d0d90af01384e06afe02090361b220df48b6e8b` |
| **Recall Checkpoint** | `mark 1/mark_4c_outputs/recall_loss_best.pth` | `c01eb4b81e4e7f1d84c7966aca56e738d87d06d404907f0bcc7c67a79ed4ec4d` |
| 3D-IRCADb Archive Set | Step 13 | `55cd6722da6c02f8365ab566603db116128e0d854782d2a174b85056f4a5ba96` |

---

## 19. Artifact & Notebook Map

All paths relative to `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`.

### 19.1 Dataset & Loader

| Purpose | Notebook / Artifact |
| :--- | :--- |
| Initial EDA | `Practice/lits_eda.ipynb`, `notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb` |
| Dataset pipeline | `Practice/dataset_pipeline.ipynb`, `Practice/unified_dataset_preparation_and_eda.ipynb` |
| Pairing forensics | `Practice/dataset_pairing_forensics.ipynb` |
| Dataset validation/promotion | `Practice/dataset_validation_and_promotion.ipynb` |
| Verified loader & overfit | `Practice/verified_manifest_loader_and_overfit_gate.ipynb` |
| Overfit gate result | `Practice/verified_loader_overfit_outputs/gate_result.json` |
| Spatial forensics (consolidated) | `notebooks/02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb` |
| Splits & characterization (consolidated) | `notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb` |
| Benchmarks & fusion (consolidated) | `notebooks/04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb` |

### 19.2 Baseline & Diagnostics (Practice)

| Stage | Notebook | Output Directory |
| :--- | :--- | :--- |
| 5-epoch smoke | `manifest_baseline_5epoch_smoke_test.ipynb` | `manifest_baseline_smoke_outputs` |
| Validation geometry | `validation_volume_mask_orientation_forensics.ipynb` | `validation_volume_forensics_outputs` |
| Patient-aware baseline | `baseline_25epoch_patient_aware_training.ipynb` + `auto_cool_continue_patient_aware_to_epoch10.ipynb` | `patient_aware_baseline_outputs` |

### 19.3 Ablations (Practice)

| Experiment | Notebook | Output Directory |
| :--- | :--- | :--- |
| Lesion-balanced sampler | `patient_lesion_balanced_sampler_ablation.ipynb` | `patient_lesion_balanced_outputs` |
| Recall-aware loss | `recall_aware_focal_tversky_loss_ablation.ipynb` | `recall_aware_loss_outputs` |
| Composite loss | `stabilized_composite_loss_ablation.ipynb` | `stabilized_composite_loss_outputs` |
| Appearance forensics | `appearance_domain_robustness_forensics.ipynb` | `appearance_domain_forensics_outputs` |
| Intensity robustness | `organ_normalized_intensity_robustness_ablation.ipynb` | `intensity_robustness_outputs` |
| Adjacent-slice 2.5D | `adjacent_slice_2_5d_context_ablation.ipynb` | `context_2_5d_outputs` |
| 3D post-processing | `validation_3d_postprocessing_ablation.ipynb` | `validation_3d_postprocessing_outputs` |
| Multi-task localization | `multitask_liver_tumor_localization.ipynb` + `multitask_liver_tumor_epoch10_continuation.ipynb` | `multitask_liver_tumor_outputs` |

### 19.4 Key Current Artifacts

| Item | Path |
| :--- | :--- |
| Dataset gate | `Practice/verified_loader_overfit_outputs/gate_result.json` |
| Strongest diagnostic checkpoint | `Practice/intensity_robustness_outputs/patient_aware_best.pth` (organ-assisted caveat) |
| Deployable-pipeline experiment checkpoint | `Practice/multitask_liver_tumor_outputs/multitask_best.pth` (best epoch 8) |
| Multi-task gate files | `multitask_localization_gate_result.json`, `multitask_continuation_status.json`, `expected_vs_actual_results.csv` |
| Multi-task detailed metrics | `multitask_history.csv`, `best_validation_patient_metrics.csv`, `best_validation_per_slice.csv`, `best_validation_size_quartiles.csv` |
| Thermal record | `multitask_thermal_log.csv` |
| Part 2 data card (dataset) | `mark 1 (part 2)/step_01_pretraining_dataset_characterization/outputs/DATASET_DATA_CARD.md` |
| Part 2 test data card | `mark 1 (part 2)/step_04.../outputs/TEST_RESULTS_DATA_CARD.md` |
| Final technical report | `mark 1 (part 2)/step_05_final_research_package/outputs/FINAL_TECHNICAL_REPORT.md` |
| External evaluation data card | `mark 1 (part 2)/step_17.../outputs/EXTERNAL_EVALUATION_DATA_CARD.md` |
| Terminal project data card | `mark 1 (part 2)/step_21.../outputs/TERMINAL_PROJECT_DATA_CARD.md` |
| LiTS provenance record | `mark 1 (part 2)/08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md` |
| Mark 1 execution blueprint | `mark 1/MARK_1_EXECUTION_BLUEPRINT.md` (diagnostic protocol + decision gates) |
| Mark 1 progress tracker | `mark 1/MARK_1_PROGRESS_TRACKER.md` (Mark 4B–4E milestones; source of §10.5) |
| Limitations & failure analysis | `mark 1 (part 2)/step_05_final_research_package/outputs/LIMITATIONS_AND_FAILURE_ANALYSIS.md` |
| Reproducibility instructions | `mark 1 (part 2)/step_05_final_research_package/outputs/REPRODUCIBILITY_INSTRUCTIONS.md` |
| UP³RE research summary | `docs/04_RESEARCH_SUMMARY.md`, `docs/RESEARCH_NOVELTY.md` |
| Research changelog / future work | `research/RESEARCH_CHANGELOG.md`, `research/notes/FUTURE_WORK.md` |
| Environment snapshot | `research/ENVIRONMENT_SNAPSHOT.md` |

> ⚠️ Do not use `Practice/next_phase_alignment_and_overfit.ipynb` (references invalid pairing generation).

---

## 20. Licensing

| Component | License |
| :--- | :--- |
| Kaggle `andrewmvd/liver-tumor-segmentation` | **CC BY-NC-ND 4.0** |
| Hugging Face CADS `0004_lits` | **CC BY-NC-SA 4.0** |
| 3D-IRCADb-01 | **CC BY-NC-ND 4.0** (public academic & non-commercial) |
| Project code/docs (README badges) | CC BY-NC 4.0 (as declared) |

> ⚠️ **Caution**: Do not redistribute raw CT volumes, derived PNG builds, converted segmentations, or packaged subsets without rechecking applicable upstream license terms. Exact terms for the owner's acquired LiTS copy were never verified — owner confirmation is mandatory before redistribution.

---

## 21. Current Status, Blockers & Next Actions

### 21.1 Current Scientific State (terminal, verified)

1. Authoritative corrected manifest preserved and verified.
2. **LiTS held-out evaluation complete** — global Dice `0.767696` but **formal model acceptance failed** (V121 minimum-patient Dice = 0).
3. **3D-IRCADb-01 external contract PASSED** with patient/small-lesion/negative-control caveats (global Dice `0.847977`).
4. External citation & license verified (`Soler2010IRCADb`, CC BY-NC-ND 4.0).
5. Manuscript evidence complete but **not submission-ready** (owner fields 0/14; Step 11 inactive by owner choice).
6. Model-improvement program (Step 00): holdout sealed; smoke gate **failed all arms** → halted for arm review.

### 21.2 Non-Negotiable Safeguards

- Do not reopen the sealed LiTS test split or Step 16 external evaluation for tuning/rerun.
- Do not tune threshold 0.70 or change maximum fusion.
- Do not use ground-truth liver masks for reported validation/test preprocessing or gating.
- Do not claim clinical readiness, universal generalization, or state-of-the-art status.
- Do not make causal claims between experiments when loader/build/sampling/loss/threshold changed.
- Keep manifest and checkpoint hashes in every result.

### 21.3 Immediate Next Actions

1. **Model improvement (Step 00)**: diagnose why all four candidates failed the 16-slice overfit canary (Dice 0.06–0.15 vs 0.80 gate) before any Step 2 training — verify optimizer/LR/architecture/ROI-crop reproducibility.
2. **Probability calibration** (frozen epoch-8 multi-task checkpoint, validation only): tumor threshold sweep `0.10–0.60`, liver thresholds `0.30–0.70`, dilation kernels `1/5/11/21/31`, raw vs liver-supported probabilities; decision tree per `understanding the project/08_current_plan.md`.
3. If calibration cannot recover V104/V116/Q1: proceed to a **two-stage predicted-liver ROI model** (predicted-liver crop at higher effective resolution).
4. **Owner-controlled items** (only if submission desired): 14 Step 11 declarations, venue selection, exact LiTS dataset terms verification.

---

## 22. UP³RE-Net / MedSegX Research Track (Novelty & Findings)

> Source: `docs/04_RESEARCH_SUMMARY.md`, `docs/RESEARCH_NOVELTY.md`, `research/` (MedSegX framework). This is a **parallel research programme** that complements the Mark 1–4E / Part 2 pipeline. Hardware: RTX 3050 Ti 4 GB laptop GPU.

### 22.1 Problem & Novel Contribution

- **Extreme class imbalance**: ~937:1 background-to-tumor (this track); standard BCE/Dice under-perform.
- **High-uncertainty regions**: tumor boundaries, small lesions — single models are overconfident at error locations.
- **Novelty gap**: no prior work combines ensemble uncertainty with per-pixel loss re-weighting in a feedback loop.
- **UP³RE-Net** = Uncertainty-Propagated Per-Pixel Reweighting Ensemble Network — a **two-stage ensemble with uncertainty feedback**:
  1. **Stage 1 (Bagging)**: train 3 MobileNetV2U-Nets on stratified bootstrap samples.
  2. **Stage 1.5 (Uncertainty)**: compute per-pixel variance across ensemble predictions.
  3. **Stage 2 (Boosting with UWACL)**: train 4th model with loss weighted by per-pixel uncertainty.
  4. **Inference**: weighted ensemble of all 4 models + per-pixel confidence map (`confidence = 1 − var/max`).

### 22.2 UWACL — Uncertainty-Weighted Adaptive Compound Loss

```
L_total = (1/N) * Σ_ij [ w_ij * (λ_bce * BCE(p_ij, y_ij) + λ_dice * Dice(p_ij, y_ij)) ]
```

- `w_ij = 1 + β * (1 - exp(-σ²_ij / τ_e))` — per-pixel weight from ensemble variance (range 1.0 → ~5.88)
- `β = 5.0`; `τ_e = 0.1 * max(0.01, 1 - 0.9 * e/total_epochs)` (dynamic temperature decay); `λ_bce = 0.5`, `λ_dice = 0.5`; `pos_weight = 10.0`
- NaN guard: skip batches exceeding 3.5 GB VRAM

### 22.3 Patent Claims (6 draft claims)

1. Method: bagging ensemble → uncertainty maps → weighted loss → boosting → combined inference
2. Weight formula: `w = 1 + β·(1 − exp(−σ²/τ))`
3. UWACL loss: weighted compound loss (BCE + Dice)
4. τ schedule: linear decay 0.1 → 0.001 over epochs
5. Mutual consistency regularization (optional future claim)
6. System: memory + processor + output interface for dual output (segmentation + confidence)

### 22.4 Model Architecture (MobileNetV2U-Net)

- Input `(B,1,256,256)`; MobileNetV2 encoder (ImageNet-pretrained, first conv adapted RGB→gray by weight averaging); custom U-Net decoder; output `(B,1,256,256)` logits.
- **Parameters**: ~6.8M (5.5M encoder + 1.3M decoder).

### 22.5 Prior-Art Distinctions (summary)

| Prior Work | Their Approach | UP³RE-Net Distinction |
| :--- | :--- | :--- |
| Two-Layer Ensemble (Cognitive Computation 2024) | Predictions as input features | Variance as **loss weights** (modulates optimization) |
| Deep Ensemble (CompBioMed 2024) | Static exponential loss | **Dynamic** weights from independent ensemble variance |
| UCTNet (Pattern Recognition 2024) | Uncertainty for architectural routing | Uncertainty for **optimization weighting** |
| UG-CEMT (WACV 2025) | Semi-supervised mean teacher | Fully supervised bagging+boosting |
| DyCON (2025) | Aleatoric self-uncertainty | **Epistemic ensemble disagreement** |

### 22.6 FAUP-Net & UWACL-v2 (research designs)

- **FAUP-Net**: Full-Architecture Uncertainty Propagation — gated skip connections on last 2 encoder blocks only (VRAM-optimized); `UncertaintyHead` (Conv3×3→1 + dropout p=0.1, <50K params).
- **UWACL-v2**: multi-scale uncertainty (1×1 / 3×3 / 7×7, learned weights `[0.4, 0.35, 0.25]`) + Laplacian edge loss (λ=0.1) + τ decay.

---

## 23. MedSegX Research Changelog — Sprint Results & Ablations

> Source: `research/RESEARCH_CHANGELOG.md`. Framework sprints: Infrastructure → Migration → Evaluation → Baseline Validation → Research Preparation.

### 23.1 Sprint 4 — Baseline Validation (2026-07-06)

| Metric | Value |
| :--- | :---: |
| Dice (3-member ensemble) | `0.8508 ± 0.0483` (4,165 val slices) |
| Member 0 / 1 / 2 Dice | `0.859` / `0.906` / `0.788` |
| Inference time | `5–10 ms/slice` |
| VRAM usage | `415 MB` (batch_size=1) |
| Parameters | `6.8M` (per model) |

### 23.2 Ablation Results (1-epoch train, 2 train volumes, 1 val volume (V104) — 781 slices)

| Experiment | Model | Loss | Ensemble | Dice |
| :--- | :--- | :--- | :---: | :---: |
| baseline | MobileNetV2UNet | combined | 1 | `0.844` |
| baseline_uwaclv1 | MobileNetV2UNet | uwacl_v1 | 1 | `0.844` |
| baseline_uwaclv2 | MobileNetV2UNet | uwacl_v2 | 1 | `0.530` |
| ensemble3 | MobileNetV2UNet | combined | 3 | `0.844` |
| ensemble3_uwaclv1 | MobileNetV2UNet | uwacl_v1 | 3 | `0.844` |
| ensemble3_uwaclv2 | MobileNetV2UNet | uwacl_v2 | 3 | `0.530` |
| faupnet | FAUPNet | combined | 1 | `0.844` |
| faupnet_uwaclv1 | FAUPNet | uwacl_v1 | 1 | `0.844` |
| faupnet_uwaclv2 | FAUPNet | uwacl_v2 | 1 | `0.442` |
| faupnet_ensemble3_uwaclv2 | FAUPNet | uwacl_v2 | 3 | `0.442` |

**Key observations**: Dice `0.844` = all-background baseline; UWACLv2 (edge loss) pushes models out of all-background equilibrium at epoch 1; FAUPNet+UWACLv2 shows conflicting early gradients (needs longer training). Tests: 139 total (120 core + 19 research), all passing.

### 23.3 Results from `results/` (Supplementary — earlier pipeline)

| Model | Dice | IoU | HD95 | NSD |
| :--- | :---: | :---: | :---: | :---: |
| MobileNetV2-UNet (Single) | `0.8787 ± 0.3265` | `0.8787 ± 0.3265` | `644.47 ± 827.20` | `0.0` |
| UP³RE Ensemble (3 members) | `0.8867 ± 0.3091` | `0.8832 ± 0.3133` | `329.80 ± 622.72` | `0.1145 ± 0.2282` |

Statistical tests (Wilcoxon, p<0.05): ensemble significantly better than best single (diff `-0.008`, p=0.0) and all members; member1 == best model (p=1.0). *Note: from the earlier pipeline; controlled gates in Steps 01–21 use the corrected build.*

---

## 24. Limitations & Failure Analysis (Final Test)

> Source: `mark 1 (part 2)/step_05_final_research_package/outputs/LIMITATIONS_AND_FAILURE_ANALYSIS.md`

- Formal model acceptance failed **solely** because minimum positive-patient Dice was below 0.01; aggregate + integrity targets passed.

### 24.1 V121 (catastrophic-patient floor failure)

| Attribute | Value |
| :--- | :---: |
| Patient Dice | `0.0000000000` |
| Truth pixels | `526` (3 connected lesions) |
| Frozen ROI containment | `100.0%` |
| Max cached fused score in truth | `0.00000000` |
| Truth pixels ≥ 0.70 | `0` |
| Interpretation | **Recognition failure**, not ROI clipping |

### 24.2 Other weak test cases

- **V120**: high recall but excessive FP volume → Dice `0.1123`.
- **V127**: only 73 tumor pixels, partial overlap → Dice `0.0826`.
- 4 of 13 positive patients scored below `0.3329`.

### 24.3 Lesion-size limitation

- Smallest train-derived volume quartile: `60.3%` detection, mean matched Dice `0.238` — markedly below larger strata (Q2 `98.00%`, Q3 `97.37%`, Q4 `100.00%`).

### 24.4 Prohibited responses

- No threshold / checkpoint / fusion / ROI / post-processing / training adjustment from test failures.
- Any follow-up model = new versioned study with a new untouched evaluation cohort.

---

## 25. Reproducibility Contract & Additional Hashes

> Source: `mark 1 (part 2)/step_05_final_research_package/outputs/REPRODUCIBILITY_INSTRUCTIONS.md`

| Artifact | SHA-256 |
| :--- | :--- |
| Manifest | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |
| Frozen policy | `52d2df856709254992ab6b59f84a213a0a00f6e67293f66665b6f258d01f2131` |
| Final acceptance contract | `12e8932d71651fd9247d841a86dfcd0fa80d64d74c4f334e29b113c28213b52d` |
| One-time test run UUID | `871d289b-bf6b-4346-978f-2df02ade26ab` |
| Step 04 evidence inventory | `9b61a09119128c7b85689762aee51617ee791a2a2b1aafc56adf185dbe6672ab` |

**Reproduce**: run `step_05_final_research_package.ipynb` from a fresh kernel (reads only sealed Part 2 outputs). **Do NOT reproduce test inference** — Step 04 ledger `rerun_allowed: false`.

### Research commands (from `docs/MANUAL_RESEARCH_VALIDATION.md`)

```powershell
# Resume baseline to epoch 5 (exact RNG checkpointing after legacy epoch-1)
.venv\Scripts\python.exe tools\train.py --config configs\experiments\research_baseline.yaml --resume models\research_validation\baseline_focal_dice\last_checkpoint.pth --epochs 5

# Validation-only evaluation (test stays locked)
.venv\Scripts\python.exe tools\evaluate.py --config configs\experiments\research_baseline.yaml --checkpoint models\research_validation\baseline_focal_dice\best_checkpoint.pth --split val --output-dir experiments\research_validation\baseline_focal_dice\val_evaluation
```

---

## 26. Result Levels & Success-Criteria Contract

> Source: `mark 1 (part 2)/07_SUCCESS_CRITERIA_PARAMETERS_AND_OUTPUTS.md`

| Level | Meaning | Permitted Next Action |
| :--- | :--- | :--- |
| `DIAGNOSTIC_COMPLETE` | Question answered; no model gate passed | Plan evidence-based diagnostic/intervention |
| `FAILED_GATE` | ≥1 mandatory target failed | Do not advance to test; diagnose/stop |
| `PARTIAL_PASS` | Most targets passed, ≥1 mandatory failed | Continue validation-only work |
| `TEMPORARY_CONTINUATION_PASS` | All temporary targets passed | Bounded validation confirmation only |
| `VALIDATION_FREEZE_PASS` | Frozen policy reproduced; all freeze targets passed | Freeze policy; request final test authorization |
| `FINAL_TEST_COMPLETE` | One-time frozen test completed | Report; no test-driven tuning |
| `FINAL_PROJECT_COMPLETE` | Test + uncertainty + failure analysis + reproducibility complete | Final paper/report handoff |

- Mark 4E = `TEMPORARY_CONTINUATION_PASS` (not final success).
- **Required per-phase outputs**: `configuration.json`, `provenance.json`, `expected_vs_actual.csv`, `gate_result.json`, `patient_metrics.csv`, slice/lesion metrics, decision dashboard PNG, localization panels, probability caches. Training adds `history.csv`, best/last checkpoints + hashes, sampler audit, thermal record.
- **Final acceptance vs aspirational targets**: formal minimum final acceptance = 6 temporary guardrails + deterministic reproducibility + no catastrophic patient failure; aspirational (research) targets = mean Dice ≥ 0.4069, V104 ≥ 0.50, V116 ≥ 0.05, Q1 ≥ 45%, pos-empty ≤ 20%, empty-FP ≤ 15%. **Aspirational targets were NOT met.**

---

## 27. Environment Snapshot

> Source: `research/ENVIRONMENT_SNAPSHOT.md` (MedSegX), `research/RESEARCH_CHANGELOG.md` (core), `mark 1 (part 2)` gates.

| Component | Value |
| :--- | :--- |
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop (4 GB VRAM, compute 8.6) |
| CUDA Driver / Runtime | 592.00 / 13.1 |
| Python | 3.11.15 (managed by `uv`) |
| torch | 2.5.1+cu124 |
| torchvision | 0.20.1 |
| numpy | 1.26.4 (pinned <2) |
| scipy | 1.17.1 / 1.14.1 |
| opencv-python | 5.0.0.93 / 4.10.0 |
| mlflow | 3.14.0 / 2.19.0 |
| streamlit | 1.58.0 |
| nibabel | 5.4.2 |
| Total packages | 99 |
| Tests | 139 (120 core + 19 research) |
| Thermal controls | Start < 84°C, emergency 90°C, 30 s cooldown polls, ≤30 min bounded wait |

**Reproduce env**: `uv sync` (or `uv pip install -r requirements.txt`) from project root; recreate venv with `uv venv .venv --python 3.11`.

---

## 28. Future Work & Research Directions

> Source: `research/notes/FUTURE_WORK.md`, `understanding the project/08_current_plan.md`.

### 28.1 Quick-Win Ideas (days, no retraining)

1. **Uncertainty-Guided Human-AI Triage** — flag top-N% highest-uncertainty volumes for radiologist review; measure review time saved.
2. **Calibration Study & Reliability Diagrams** — temperature/Platt scaling; ECE, Brier, MCE.
3. **LLM-Augmented Clinical Reporting** — segmentation + tumor-burden analytics → structured radiology reports.
4. **Knowledge Distillation** — distill 3–4 model ensemble into a single lightweight student.
5. **Low-Data Regime Study** — train with 10/25/50/100% of the 104 train volumes.

### 28.2 Medium-Term (1–2 weeks)

- Failure-mode & interpretability analysis (Grad-CAM, error taxonomy).
- Cross-dataset generalization (3D-IRCADb already done in Steps 13–17; consider CHAOS/HCC-TACE-Seg as second external cohort).
- Model zoo expansion (U-Net, Attention U-Net, U-Net++, DeepLabV3+, TransUNet).

### 28.3 Major Architecture Overhaul (weeks)

- **FAUP-Net** full-architecture uncertainty propagation (encoder/decoder block uncertainty, gated skip connections, multi-scale disagreement aggregator, UWACL-v2) — flagship novelty candidate for MICCAI/TMI/MedIA.
- Self-supervised pre-training (MAE / contrastive); TTA + 3D sliding window (+0.5–2% Dice typical).

### 28.4 Immediate scientific next steps (from Part 2 current plan)

1. **Step 00 diagnosis**: determine why all four smoke-gate candidates failed the 16-slice overfit canary (Dice 0.06–0.15 vs 0.80 gate) before any Step 2 training.
2. **Probability calibration** (frozen epoch-8 multi-task checkpoint, validation-only): tumor thresholds `0.10–0.60`, liver thresholds `0.30–0.70`, dilation kernels `1/5/11/21/31`; decision tree in `MARK_1_EXECUTION_BLUEPRINT.md`.
3. If calibration cannot recover V104/V116/Q1: **two-stage predicted-liver ROI model** at higher effective resolution.

---

*End of consolidated evaluation data. Generated from repository documentation: `README.md`, `dataset.md`, `docs/*`, `understanding the project/*`, `mark 1/*`, `mark 1 (part 2)/*` (incl. step outputs), `Practice/*`, `configs/*`, `results/*`, `research/*`.*

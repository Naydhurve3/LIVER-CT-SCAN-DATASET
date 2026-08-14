# LiTS Liver-Tumour Segmentation Dataset — Master Reference

This document consolidates every verified detail about the local LiTS cohort that has been extracted across all project folders (`mark 1\`, `mark 1 (part 2)\`, `Practice\`, `understanding the project\`, and the dataset staging/build directories). All numerical values below are derived directly from the authoritative corrected build (`build_corrected_20260713_214847_v2`), its slice manifest, and machine-readable JSON audit artifacts.

> **Source of Truth**: The corrected slice manifest (`slice_manifest.csv`). If any table or section in this document disagrees with the manifest, the manifest and its cryptographic SHA-256 hash take absolute precedence.

---

## 1. Identity and Provenance

- **Dataset Name**: Liver Tumor Segmentation Benchmark (LiTS-17) — 3D Abdominal CT Volumes with Liver & Tumour Annotations.
- **Challenge Origin**: LiTS (ISBI 2017 / MICCAI 2017–2018), CodaLab Competition ID `17094`.
- **Benchmark Publication**: Bilic P, et al. *The Liver Tumor Segmentation Benchmark (LiTS).* Medical Image Analysis. 2023;84:102680. doi:[10.1016/j.media.2022.102680](https://doi.org/10.1016/j.media.2022.102680).
- **Local Cohort**: The **131 annotated primary CT volumes and segmentation pairs**, numbered `0` through `130`.
- **Local Canonical Build ID**: `build_corrected_20260713_214847_v2`
- **Build Directory**: `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- **Manifest Location**: `manifests\slice_manifest.csv`
- **Manifest SHA-256 Hash**: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- **Manifest Grain**: One axial slice per row (**58,638 total rows**, 27 columns).
- **Voxel Annotation Values**:
  * `0 = Background` (outside organ of interest)
  * `1 = Liver Parenchyma`
  * `2 = Tumour / Lesion`
- **Modelling Target**: Tumour-only binary foreground ($1 = \text{Tumour}$, $0 = \text{Background}$); Liver/Organ mask is stored separately for ROI bounding-box extraction.
- **Sealed Test Set Boundary**: The project's internal `test` split is a **sealed 14-volume holdout carved from the 131 annotated training volumes**. It is **not** the official 70-volume unlabelled LiTS challenge test set.

### 1.1 Source Acquisition Record

| Source Mirror | Volume ID Range | Volume Count | Raw Acquired Format |
| :--- | :---: | :---: | :--- |
| Kaggle `andrewmvd/liver-tumor-segmentation` | `0–50` | 51 | NIfTI (`.nii`) volume + segmentation pairs |
| Kaggle `andrewmvd/liver-tumor-segmentation-part-2` | `51–69`, `100` | 20 | NIfTI (`.nii`) pairs (extracted from full archive) |
| Hugging Face CADS `0004_lits` | `70–99`, `101–130` | 60 | Gzip-compressed NIfTI (`.nii.gz` decompressed to `.nii`) |
| **Total Primary Cohort** | **`0–130`** | **131** | **131 Primary CT Volumes + 131 Multiclass Segmentations** |

### 1.2 Historical Image Working Copies

From `Dataset\Liver\00_source_registry\source_registry.json` (SHA-256: `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173`):

| Source Folder Name | Absolute Local Directory Path | Contents & Inventory |
| :--- | :--- | :--- |
| `lits-png` | `D:\DATA SCIENCE AND ANALYTICS\Dataset\lits-png\dataset_6\dataset_6` | 58,638 image PNGs, 15,868 liver masks, 15,817 lesion masks |
| `LiTS_masks` | `D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks` | 58,638 tumor-mask PNGs ($256\times 256$) |
| `Liver Img Dataset` | `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset` | 58,638 legacy image PNGs ($512\times 512$, unverified spatial alignment) |

> [!IMPORTANT]  
> These folders represent historical working extractions of the **same 131-volume cohort** and do not contain independent patient data. The standardized NIfTI $\to$ `build_corrected_20260713_214847_v2` pipeline is the sole canonical data source.

---

## 2. Directory Layout & Data Physical Architecture

### 2.1 Storage Root `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\`

```
D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\
├── 00_source_registry/           <- Inventory catalog (source_registry.json)
├── 01_raw_authoritative/         <- Master NIfTI files (.nii) & acquisition download logs
│   ├── volumes/                  <- volume-0.nii through volume-130.nii (131 files)
│   └── segmentations/            <- segmentation-0.nii through segmentation-130.nii (131 files)
├── 02_staging/                   <- Derived PNG staging builds
│   └── build_corrected_20260713_214847_v2/  <- CANONICAL AUTHORITATIVE BUILD
├── 03_derived_256/               <- 256x256 pre-cropped ROI & cache artifacts
├── 04_manifests/                 <- Global CSV manifests & checksum registers
├── 05_splits/                    <- Train, validation, and test split manifests
├── 06_audits/                    <- Automated integrity JSON logs & verification reports
├── 07_training_cache/            <- High-speed PyTorch array caches (.npz / .pt)
└── 99_quarantine/                <- Quarantined slices / anomalous data
```

### 2.2 Corrected Staging Build `build_corrected_20260713_214847_v2` Layout

```
build_corrected_20260713_214847_v2/
├── images/                       <- 131 volume subdirectories (v000 to v130), 58,638 PNGs (256x256)
├── organ_masks/                  <- 131 volume subdirectories, 58,638 binary organ PNGs
├── tumor_masks/                  <- 131 volume subdirectories, 58,638 binary tumor PNGs
├── manifests/
│   ├── slice_manifest.csv        <- Master slice-level manifest (58,638 rows, 27 columns)
│   ├── eda_nonspatial_manifest.csv
│   ├── eda_spatial_manifest.csv
│   └── quarantine_manifest.csv
├── splits/
│   ├── train_slices.csv          <- 40,667 rows
│   ├── val_slices.csv            <- 10,685 rows
│   ├── test_slices.csv           <- 7,286 rows
│   ├── train_volumes.txt / val_volumes.txt / test_volumes.txt
│   └── split_hashes.json
├── audits/
│   ├── strict_validation_summary.json
│   ├── volume_summary.csv
│   └── split_summary.csv
├── spatial_reviews/             <- 131 per-volume review sheets + 7 batch contact sheets
├── dataset_version.json
├── dataset_readiness.json
└── dataset_card.txt
```

---

## 3. Cohort Statistics & Patient-Disjoint Splits

### 3.1 Global Cohort Metrics

* **Total CT Volumes (Patients)**: `131`
* **Total Axial Slices**: `58,638`
* **Organ-Positive Slices**: `19,156` (`32.67%`)
* **Tumour-Positive Slices**: `7,169` (`12.23%`)
* **Organ Pixel Count**: `87,806,713`
* **Tumour Pixel Count**: `4,615,891`
* **In-Plane Spatial Orientation Fixes**: `47` volumes required a $180^\circ$ rotation fix (Volumes `83–99` and `101–130`). `84` volumes used identity.
* **Strict Validation Pass Rate**: `100%` (`0` failures across `175,914` image/mask files).

### 3.2 Canonical Patient-Disjoint Split Composition

Splits were created strictly by volume ID to prevent data leakage between adjacent axial slices of the same patient.

| Split Name | Volume ID Range | Volume Count | Total Slice Count | Organ-Positive Slices | Tumour-Positive Slices | Tumour-Positive Slices % | Mean Volume Tumour Burden % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `0 – 103` | `104` | `40,667` | `13,382` | `4,930` | `12.12%` | `0.1003%` |
| **Validation** | `104 – 116` | `13` | `10,685` | `3,612` | `1,042` | `9.75%` | `0.0874%` |
| **Test (Sealed)** | `117 – 130` | `14` | `7,286` | `2,162` | `1,197` | `16.43%` | `0.3071%` |
| **Total Cohort** | **`0 – 130`** | **131** | **58,638** | **19,156** | **7,169** | **12.23%** | **0.1218%** |

* **Tumour-Positive Patients**: 96 in Train, 9 in Validation (the 9 define the Mark 4E validation target cohort).
* **Test Holdout Lock**: Test slices/images are **strictly sealed** and have never been opened for EDA, threshold search, checkpoint selection, or loss tuning.

---

## 4. Slice Manifest Schema (27 Columns)

The master manifest (`manifests\slice_manifest.csv`) contains 58,638 rows structured as follows:

| Column Name | Data Type | Description / Standardized Values |
| :--- | :---: | :--- |
| `sample_id` | String | Unique slice ID in format `v{NNN}_s{NNNN}` (e.g. `v000_s0000`) |
| `volume_id` | Integer | Patient volume ID from `0` to `130` |
| `slice_index` | Integer | Zero-indexed axial slice number within volume (`0` to $N-1$) |
| `image_path` | String | Relative path to image PNG (`images/v000/s0000.png`) |
| `organ_mask_path` | String | Relative path to organ mask PNG (`organ_masks/v000/s0000.png`) |
| `tumor_mask_path` | String | Relative path to tumour mask PNG (`tumor_masks/v000/s0000.png`) |
| `source_volume_path` | String | Absolute path to raw source NIfTI volume (`01_raw_authoritative/volumes/volume-0.nii`) |
| `source_segmentation_path` | String | Absolute path to raw NIfTI segmentation |
| `source_volume_sha256` | String | Cryptographic SHA-256 hash of raw NIfTI volume |
| `source_segmentation_sha256` | String | Cryptographic SHA-256 hash of raw NIfTI segmentation |
| `transform_applied` | String | Applied spatial orientation transform: `identity` or `rot180` |
| `image_width` | Integer | Image pixel width (`256`) |
| `image_height` | Integer | Image pixel height (`256`) |
| `organ_pixels` | Integer | Count of non-zero organ pixels in slice mask |
| `tumor_pixels` | Integer | Count of non-zero tumour pixels in slice mask |
| `organ_present` | Boolean | `True` if `organ_pixels > 0`, else `False` |
| `tumor_present` | Boolean | `True` if `tumor_pixels > 0`, else `False` |
| `automatic_integrity_pass` | Boolean | `True` for all 58,638 rows (no corruption) |
| `verification_status` | String | `verified` for all rows |
| `exclusion_reason` | String | Empty string (`""`) — 0 rows excluded |
| `build_id` | String | `build_corrected_20260713_214847_v2` |
| `preprocessing_profile` | String | `HU[-160,240]_bilinear_image_nearest_mask_256` |
| `mask_operation_order` | String | `derive_resize_nearest_then_transform_256` |
| `split` | String | Partition assignment: `train`, `val`, or `test` |
| `manual_spatial_status` | String | `approved` for all rows |
| `eda_nonspatial_ready` | Boolean | `True` |
| `eda_spatial_ready` | Boolean | `True` |

---

## 5. Spatial Geometry & Physical Voxel Dimensions

All raw NIfTI scans originate at $512\times 512$ in-plane resolution. Physical volume calculation follows:
$$V_{\text{physical}} = \text{Pixel Count} \times \Delta x \times \Delta y \times \Delta z \quad (\text{mm}^3)$$

### 5.1 Volume Geometry Profile (117 Train + Validation Scans)

| Feature Parameter | Train (104 Vols) Median | Validation (13 Vols) Median | Observed Cohort Range |
| :--- | :---: | :---: | :---: |
| **Slices per Volume ($N_z$)** | `263` | `816` | `[74, 987]` |
| **In-Plane Spacing ($\Delta x, \Delta y$)** | `0.772 mm` | `0.781 mm` | `[0.557, 1.000] mm` |
| **Slice Thickness ($\Delta z$)** | `1.000 mm` | `0.800 mm` | `[0.450, 5.000] mm` |
| **Field-of-View X/Y ($\text{FOV}_{xy}$)** | `395.5 mm` | `399.9 mm` | `[285.2, 512.0] mm` |
| **Field-of-View Z ($\text{FOV}_z$)** | `469.5 mm` | `635.6 mm` | `[74.0, 808.0] mm` |
| **Physical Liver Volume** | `1,636 mL` | `1,575 mL` | `[583, 3,341] mL` |
| **Physical Tumour Volume** | `14.3 mL` | `3.6 mL` | `[0.0, 968.6] mL` |
| **Predicted Liver ROI Crop Area Ratio** | `0.420` | `0.427` | `[0.220, 0.606]` |
| **ROI Crop Height** | `173 px` | — | `[114, 235] px` |
| **ROI Crop Width** | `156 px` | — | `[117, 207] px` |

### 5.2 Frozen Liver ROI Extraction Protocol

To focus segmentation on the liver parenchyma and eliminate extraneous background tissue, a **2-Stage ROI extraction policy** is applied:
1. **Liver Score Threshold**: $0.50$ probability threshold from Stage-1 Liver Net.
2. **Component Filtering**: Select `largest_3d` connected volume component.
3. **Spatial Padding**: Add $+16$ pixels padding in $X$ and $Y$ dimensions around bounding box $[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$.
4. **ROI Rescaling**: Standardized to $256\times 256$.
5. **Coverage Verification**: **100%** containment of all tumor pixels across Train and Validation (`152,763 / 152,763` pixels in `V116`).

---

## 6. 3D Lesion Morphology & Tumor Distribution Bins

Across the cohort, **845 distinct 3D 6-connected lesion components** were identified and analyzed.

| Lesion Metric | Value / Median | Distribution Notes |
| :--- | :---: | :--- |
| **Train 3D Lesions** | `637` | Components isolated via 6-connectivity |
| **Validation 3D Lesions** | `208` | Components isolated via 6-connectivity |
| **3D Lesion Volume** | `0.383 mL` | Range: `[0.00035, 968.60] mL` (Extremely right-skewed) |
| **Equivalent Spherical Diameter** | `9.01 mm` | Range: `[0.87, 122.70] mm` |
| **Axial Span ($N_z$ slices)** | `7 slices` | Median depth |
| **Lesions per Patient** | `4 lesions` | Max: `70 lesions` in a single patient |
| **Max Tumour-to-Liver Ratio** | `0.314` | Severe multifocal disease |

### 6.1 Train-Derived Lesion Stratification Bins (`train_derived_lesion_bins.json`)

To evaluate model performance without validation data leakage, lesion size quartiles were fit strictly on training data:

| Stratum / Bin Name | Lesion Volume Range ($V_L$) | Spherical Diameter Range ($d_e$) | Patient Lesion Count | Patient Tumour Vol |
| :--- | :---: | :---: | :---: | :---: |
| **Q1 (Very Small / Small)** | $(-\infty, 0.173\text{ mL}]$ | $(-\infty, 6.92\text{ mm}]$ | $(-\infty, 1.0]$ | $(-\infty, 1.96\text{ mL}]$ |
| **Q2 (Medium-Small)** | $(0.173, 0.673\text{ mL}]$ | $(6.92, 10.87\text{ mm}]$ | $(1.0, 2.5]$ | $(1.96, 14.32\text{ mL}]$ |
| **Q3 (Medium-Large)** | $(0.673, 3.944\text{ mL}]$ | $(10.87, 19.60\text{ mm}]$ | $(2.5, 8.0]$ | $(14.32, 42.14\text{ mL}]$ |
| **Q4 (Large / Massive)** | $(3.944, +\infty\text{ mL})$ | $(19.60, +\infty\text{ mm})$ | $(8.0, +\infty)$ | $(42.14, +\infty\text{ mL})$ |

---

## 7. Radiometric Appearance & Hounsfield Unit (HU) Profile

Raw attenuation values from NIfTI files were evaluated across liver parenchyma, tumor tissue, and background.

$$\text{HU} = 1000 \times \frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}} - \mu_{\text{air}}}$$

### 7.1 Intensity Statistics

| Structure / Contrast Metric | Train Median | Validation Median | Standardized Shift ($\Delta\sigma$) |
| :--- | :---: | :---: | :---: |
| **Tumour HU Median** | `+67 HU` | `+59 HU` | $-0.16\sigma$ |
| **Tumour HU IQR** | `36 HU` | `39 HU` | $-0.04\sigma$ |
| **Liver HU Median** | `+99 HU` | `+106 HU` | $+0.36\sigma$ |
| **Tumour-minus-Liver Contrast** | `-34 HU` | `-44 HU` | $-0.55\sigma$ (Val lesions are more hypodense) |
| **Robust Contrast-to-Noise Ratio (CNR)** | `-1.52` | `-1.88` | $-0.46\sigma$ |
| **Background HU Median** | `-887 HU` | `-908 HU` | $-0.43\sigma$ |

### 7.2 Standard Preprocessing & Normalization Pipeline

```
Raw NIfTI (.nii) ──► Clip HU [-160, +240] ──► Min-Max Scale [0, 1] ──► Bilinear Resize to 256x256 PNG
```
* **Broad Abdominal Window**: `[-160, +240] HU`
* **Image Interpolation**: Bilinear
* **Mask Interpolation**: Nearest-Neighbour (preserves binary integer labels $0, 1$)
* **Stored Image Dimensions**: $256\times 256$ pixels, 8-bit grayscale PNG.

---

## 8. Data-Integrity Forensics & Audit History

Prior to model training, multi-stage data audits resolved major historical dataset anomalies:

| Identified Anomaly / Issue | Root Cause Analysis | Verified Resolution & Corrective Action |
| :--- | :--- | :--- |
| **47 Volume Orientation Flips** | Image and mask matrices had $180^\circ$ in-plane mismatch in Kaggle/HF imports | Applied $180^\circ$ spatial rotation matrix to Volumes `83–99` and `101–130`. |
| **Slice Offset Discrepancy** | Investigated axial slice indexing shifts ($z$-offset) | Determined optimal slice offset was $0$; problem was orientation, not offset. |
| **4x Tumour Burden Error** | Legacy code used $512\times 512$ denominator ($262,144\text{ px}$) on $256\times 256$ masks | Recomputed metrics using $256\times 256$ native mask area ($65,536\text{ px}$), adjusting train burden from `0.0251%` to `0.1003%`. |
| **30,105 Mask Values >255** | Multi-instance label encoding overflow in legacy exporter | Applied `instance mod 256` matching all 58,638 rows. |
| **Tumour-Positive Slice Count Mismatch** | Legacy CSV listed 7,114 positive slices | Verified true slice manifest count of **`7,169`** positive slices. |

---

## 9. JSON Audit Schemas & Cryptographic State Registers

The project relies on machine-readable JSON registers to validate execution state and reproducibility.

### 9.1 `dataset_version.json` Schema & Values
* **`build_id`**: `"build_corrected_20260713_214847_v2"`
* **`status`**: `"verified_eda_ready"`
* **`counts`**: `{"volumes": 131, "slices": 58638, "organ_positive_slices": 19156, "tumor_positive_slices": 7169, "organ_pixels": 87806713, "tumor_pixels": 4615891}`
* **`transforms.rot180`**: `[83..99, 101..130]` (47 volumes)
* **`transforms.identity`**: `[0..82, 100]` (84 volumes)

### 9.2 `dataset_readiness.json` State Gate
* **`automatic_gates_pass`**: `true`
* **`strict_validation_failures`**: `0`
* **`spatial_review_approved_volumes`**: `131`
* **`nonspatial_eda_ready`**: `true`
* **`spatial_eda_ready`**: `true`
* **`training_ready`**: `true` *(Frozen loader validation passed; 16-slice overfit gate passed at hard micro-Dice `0.8163` ≥ `0.80`; 5-epoch manifest baseline smoke passed; `VerifiedManifestDataset` wired into the experiment builder)*

> **Training-Readiness Note (2026-08-12):** The two gates required by the readiness artifact — manifest-driven loader validation and the 16-slice overfit Dice `>= 0.80` check — were passed (`Practice/verified_loader_overfit_outputs/gate_result.json`: loader `true`, overfit `true`, best hard micro-Dice `0.8163059889`; 6/6 manifest-loader unit tests pass). The training path now loads data exclusively from the verified manifest via `VerifiedManifestDataset` (`create_manifest_dataloaders` in `src/framework/data/manifest_dataset.py`, routed through `build_experiment_loaders` in `src/framework/experiment.py`). Run a manifest-based experiment with `configs/experiments/research_manifest_baseline.yaml`. The sealed test split remains locked (`allow_test=False`), and training run manifests record `test_set_accessed: false` plus split/checkpoint hashes.

---

## 10. External Evaluation Benchmark: 3D IRCADb-01

To evaluate out-of-domain model generalization without risking data leakage on the sealed test set, the **3D IRCADb-01** public dataset was integrated (`step_12` through `step_17` in `mark 1 (part 2)`).

* **Dataset Source**: IRCAD (Research Institute against Digestive Cancer), France.
* **Volume Count**: **20 anonymized abdominal CT scans** (`2,827` total axial slices).
* **Lesion Annotations**: 3D manual segmentations of liver, hepatic tumors, vessels, and surrounding organs.
* **Standardized Ingestion Pipeline**:
  * Extracted from DICOM to NIfTI.
  * Standardized to identical `256 x 256` spatial resolution using HU `[-160, 240]` windowing.
  * Preserves un-tuned external evaluation parity for cross-center validation.

---

## 11. Model Development & Experimental Evidence (`mark 1\`)

| Phase Name | Notebook File Path | Primary Objective / Gate | Outcome / Key Results |
| :--- | :--- | :--- | :--- |
| **Mark 1** | `mark 1\mark_1_probability_contrast_localization_diagnostic.ipynb` | Baseline diagnostic & probability distribution | Identified hypodense lesion suppression |
| **Mark 2** | `mark 1\mark_2_roi_multiwindow_feasibility.ipynb` | Multi-window & predicted-liver ROI feasibility | Validated 2-stage ROI crop workflow |
| **Mark 3** | `mark 1\mark_3_two_stage_multiwindow_overfit.ipynb` | 2-stage deterministic overfit gate | Achieved hard micro-Dice **`0.900557`** (17 epochs); 100% tumor containment |
| **Mark 4 / 4B** | `mark 1\mark_4_two_stage_validation_smoke.ipynb` | 5-epoch ROI validation smoke training | Revealed high positive predicted-empty rate (`36.85%`) |
| **Mark 4C / 4D** | `mark 1\mark_4d_metric_reconciliation_v116_diagnostic.ipynb` | Recall-loss ablation & patient reconciliation | Diagnostic of `V116` hypodense lesion localization failure |
| **Mark 4E** | `mark 1\mark_4e_checkpoint_fusion_validation.ipynb` | Checkpoint fusion validation gate | **PASSED ALL 6 VALIDATION TARGETS** using pixelwise max fusion ($\text{threshold} = 0.70$) |

### Mark 4E Validation Gate Results (9 Tumour-Positive Validation Patients)

$$\text{Fused Probability} = \max(P_{\text{control}}, P_{\text{recall\_loss}})$$

| Benchmark Metric | Target Requirement | Mark 4E Actual Result | Status Gate |
| :--- | :---: | :---: | :---: |
| **Mean Positive-Patient Dice** | $\ge 0.3329$ | **`0.377087`** | **PASS** |
| **V104 Patient Dice** | $\ge 0.0500$ | **`0.116627`** | **PASS** |
| **V116 Patient Dice** | $\ge 0.0100$ | **`0.010474`** | **PASS** |
| **Q1 Small Lesion Detection Rate** | $\ge 35.0\%$ | **`50.57%`** | **PASS** |
| **Positive Predicted-Empty Rate** | $\le 35.0\%$ | **`27.45%`** | **PASS** |
| **Empty-Slice False Positive Rate** | $\le 20.0\%$ | **`5.55%`** | **PASS** |

---

## 12. Licensing & Redistribution Terms

- Kaggle `andrewmvd/liver-tumor-segmentation`: **CC BY-NC-ND 4.0**
- Hugging Face CADS `0004_lits`: **CC BY-NC-SA 4.0**
- 3D IRCADb-01: **Public Academic & Non-Commercial License**

> [!CAUTION]  
> Do not redistribute raw CT volumes, derived PNG builds, converted segmentations, or packaged subsets without rechecking applicable upstream license terms. Maintain non-commercial attribution at all times.

---

## 13. Cryptographic Hashes & Verification Fingerprints

| Target File / Artifact | Relative Path | Cryptographic SHA-256 Hash |
| :--- | :--- | :--- |
| **Master Slice Manifest** | `manifests\slice_manifest.csv` | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |
| **Train Slices CSV** | `splits\train_slices.csv` | `6b0e2753dbbf780383fdf25f4421f32c496752265d83a348fbda38d2291492cc` |
| **Train Volumes TXT** | `splits\train_volumes.txt` | `6183e84acb5a30fd04a38875f54c6a02b4f64f588725f236ff3984ef4450a62d` |
| **Validation Slices CSV** | `splits\val_slices.csv` | `4b3d6abd06823239dda131add2eb407d48078fcd17b931acfbff7128425312b1` |
| **Validation Volumes TXT** | `splits\val_volumes.txt` | `a28fea675229371093748adb90e9c1d23dd2a5699fd6f833a902ee4681fdb9ad` |
| **Test Slices CSV (Sealed)** | `splits\test_slices.csv` | `6034134006793c9f5a1392aec51c3e44eb47b54d6ab2e74aca11c8166e1bca0a` |
| **Test Volumes TXT (Sealed)**| `splits\test_volumes.txt` | `830f8ded37afeb01347c5f222d91d253ac11cad8bc9621ce8cb04b0ef6b8d154` |
| **Corrected `dataset_version.json`** | `dataset_version.json` | `fb39183db4d59b8010dbb77426e4ba2166a1316c6b2da6925996a41739fd22bf` |
| **Corrected `dataset_readiness.json`** | `dataset_readiness.json` | `2dad7f66e37edc57be4e27f1e68707f2d48a5bd432f962ba703734db7bf92bc8` |
| **Source Registry JSON** | `00_source_registry\source_registry.json` | `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173` |

---

## 14. Canonical Reuse Statement

> The project utilizes the 131 annotated LiTS CT volumes numbered 0–130. Volumes 0–69 and 100 were acquired from the Kaggle LiTS mirrors, while volumes 70–99 and 101–130 were acquired from the Hugging Face CADS `0004_lits` mirror and decompressed from `.nii.gz` to `.nii`. The cohort was standardized into canonical build `build_corrected_20260713_214847_v2`; its authoritative slice manifest has SHA-256 `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`. The project's internal test split is a sealed holdout from these 131 annotated volumes, not the official LiTS challenge test set.

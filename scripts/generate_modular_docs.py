import os

os.makedirs("docs", exist_ok=True)

# 1. docs/DATA_PROVENANCE_AND_ACQUISITION.md
doc1 = """# Dataset Identity, Provenance, and Source Acquisition Specification

[![Modality](https://img.shields.io/badge/Modality-3D%20Abdominal%20CT-green.svg)]()
[![Volumes](https://img.shields.io/badge/Volumes-131-orange.svg)]()
[![Slices](https://img.shields.io/badge/Slices-58%2C638-purple.svg)]()
[![Build ID](https://img.shields.io/badge/Build%20ID-build__corrected__20260713__214847__v2-success.svg)]()
[![Manifest SHA-256](https://img.shields.io/badge/SHA--256-575a6fc391...-blue.svg)]()

> **Primary Source Reference**: LiTS-17 Liver Tumor Segmentation Benchmark (ISBI 2017 / MICCAI 2017–18, CodaLab Competition 17094).  
> **Canonical Staging Directory**: `D:\\DATA SCIENCE AND ANALYTICS\\Dataset\\Liver\\02_staging\\build_corrected_20260713_214847_v2`  
> **Executable Notebook Reference**: [`notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb`](../notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb)  
> **Phase Handoff Reference**: [`mark 1 (part 2)/08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md`](../mark%201%20(part%202)/08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md)

---

## 1. Complete Acquisition Pipeline & Data Flow

The cohort comprises **131 primary 3D abdominal CT scans** ($58,638$ axial slices) collected from three official mirror sources. The complete pipeline transforms raw DICOM/NIfTI archives into a cryptographically verified staging build.

```mermaid
flowchart TD
    subgraph Data Sources & Mirrors
        A1["Kaggle Mirror Part 1<br>(andrewmvd/liver-tumor-segmentation)<br>Volumes 0 – 50 (51 Pairs)"]
        A2["Kaggle Mirror Part 2<br>(andrewmvd/liver-tumor-segmentation-part-2)<br>Volumes 51 – 69, 100 (20 Pairs)"]
        A3["Hugging Face CADS Mirror<br>(huggingface/CADS-dataset 0004_lits)<br>Volumes 70 – 99, 101 – 130 (60 Pairs .nii.gz)"]
    end

    subgraph Acquisition & Extraction
        B1["Direct NIfTI (.nii) Ingestion"]
        B2["Gzip Decompression (.nii.gz -> .nii)"]
    end

    subgraph Spatial Forensics & Quality Audit
        C1["Raw 512x512 NIfTI Inspection<br>(131 Volumes, 58,638 Slices)"]
        C2{"In-Plane Orientation Audit"}
        C3["Rot180° Transformation Matrix<br>(Volumes 83–99 & 101–130)"]
        C4["Identity Matrix<br>(Volumes 0–82 & 100)"]
    end

    subgraph Standardized Preprocessing
        D1["Clip Hounsfield Units [-160, +240] HU"]
        D2["Linear Min-Max Normalization to [0, 1]"]
        D3["Bilinear Rescale Images to 256x256 PNG"]
        D4["Nearest-Neighbour Rescale Masks to 256x256 PNG"]
    end

    subgraph Canonical Staging Build
        E1["build_corrected_20260713_214847_v2"]
        E2["slice_manifest.csv (58,638 rows)<br>SHA-256: 575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B2 --> B1
    B1 --> C1 --> C2
    C2 -- "47 Flipped Volumes" --> C3 --> D1
    C2 -- "84 Identity Volumes" --> C4 --> D1
    D1 --> D2 --> D3 --> E1
    D1 --> D2 --> D4 --> E1
    E1 --> E2
```

---

## 2. Acquisition Source Mapping & Volume Distribution

| Source Mirror | Volume ID Range | Volume Count | Raw Acquired Format | Processing & Transformation | Verification Status |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **Kaggle LiTS Part 1** | `0 – 50` | `51` | NIfTI (`.nii`) Pairs | Identity Transform ($I$) | Verified & Approved |
| **Kaggle LiTS Part 2** | `51 – 69`, `100` | `20` | NIfTI (`.nii`) Pairs | Volumes `51–69` Rot180°, `100` Identity | Verified & Approved |
| **Hugging Face CADS (`0004_lits`)** | `70 – 99`, `101 – 130` | `60` | Compressed NIfTI (`.nii.gz`) | Gzip Decompressed $\\to$ Rot180° applied | Verified & Approved |
| **Canonical Primary Cohort** | **`0 – 130`** | **131** | **131 Volume/Seg Pairs** | **Canonical Staging Build** | **100% Strict Audit Pass** |

---

## 3. Mathematical Foundations & Processing Equations

### 3.1 Radiometric Hounsfield Unit (HU) Scaling
Raw CT attenuation coefficients ($\mu$) are converted to Hounsfield Units relative to water ($\mu_{\\text{water}}$) and air ($\mu_{\\text{air}}$):

$$\\text{HU} = 1000 \\times \\frac{\\mu_{\\text{tissue}} - \\mu_{\\text{water}}}{\\mu_{\\text{water}} - \\mu_{\\text{air}}}$$

To extract parenchymal contrast without bone or air interference, intensity clipping and linear scaling are applied:

$$I_{\\text{norm}}(x, y) = \\text{clamp}\\left( \\frac{\\text{HU}(x, y) - (-160)}{240 - (-160)}, 0.0, 1.0 \\right) = \\text{clamp}\\left( \\frac{\\text{HU}(x, y) + 160}{400}, 0.0, 1.0 \\right)$$

Post-windowing, intensity values are quantized into 8-bit grayscale space $[0, 255]$ for PNG storage:

$$I_{8\\text{-bit}}(x, y) = \\lfloor 255 \\times I_{\\text{norm}}(x, y) \\rceil$$

---

### 3.2 In-Plane Spatial Transformation Matrix ($T_{\\text{rot180}}$)
Audit discovered that **47 volumes** exhibited a $180^\\circ$ in-plane orientation flip. The planar transformation matrix applied to 2D slice coordinates $(x, y)$ is defined as:

$$\\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} = \\mathbf{T}_{\\text{rot180}} \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} -1 & 0 & W - 1 \\\\ 0 & -1 & H - 1 \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}$$

For $256 \\times 256$ spatial dimension ($W = H = 256$):

$$x' = 255 - x, \\quad y' = 255 - y$$

---

### 3.3 Physical Volume & Reconciled Tumor Burden Equations
Physical volume calculations integrate voxel pixel counts across slice thickness ($\\Delta z$) and in-plane voxel spacing ($\\Delta x, \\Delta y$):

$$V_{\\text{physical}} = \\left( \\sum_{z=1}^{N_z} \\sum_{y=1}^{H} \\sum_{x=1}^{W} M(x, y, z) \\right) \\times \\Delta x \\times \\Delta y \\times \\Delta z \\quad (\\text{mm}^3)$$

$$\\text{Volume in mL} = \\frac{V_{\\text{physical}}}{1000}$$

Reconciled tumor burden percentage uses true native $256 \\times 256$ mask area ($65,536\\text{ px}$):

$$\\text{Tumor Burden } (\\%) = \\frac{\\sum_{i=1}^{N_z} \\text{Tumor Pixels}_i}{N_z \\times 256 \\times 256} \\times 100$$

* **Correction Note**: Legacy scripts calculated burden using a $512 \\times 512$ pixel denominator ($262,144\\text{ px}$), artificially deflating burden values by $4\\times$. Corrected native calculation adjusts mean train tumor burden from `0.0251%` to **`0.1003%`**.

---

## 4. Historical Working Directories & Inventory Registries

From `Dataset\\Liver\\00_source_registry\\source_registry.json` (SHA-256: `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173`):

| Working Directory Name | Absolute Local Directory Path | Contained Files & Inventory | Operational Role & Guidance |
| :--- | :--- | :--- | :--- |
| `lits-png` | `Dataset\\lits-png\\dataset_6\\dataset_6` | 58,638 image PNGs, 15,868 organ masks, 15,817 lesion masks | Historical Working Directory |
| `LiTS_masks` | `Dataset\\LiTS_masks` | 58,638 tumor-mask PNGs ($256 \\times 256$) | Preserved Label Source |
| `Liver Img Dataset` | `Dataset\\Liver Img Dataset` | 58,638 legacy image PNGs ($512 \\times 512$) | **Superseded** (Unverified Alignment) |

> [!CAUTION]  
> The directories above represent historical extractions of the **same 131 patients**. Do not treat them as additional independent datasets.

---

## 5. Slice Manifest Schema & Cryptographic Checksums

The master manifest (`manifests\\slice_manifest.csv`) contains 58,638 rows structured across 27 fields.

```
Key Manifest Fields:
sample_id | volume_id | slice_index | image_path | organ_mask_path | tumor_mask_path | 
transform_applied | organ_pixels | tumor_pixels | organ_present | tumor_present | split
```

### Cryptographic Fingerprint Registry:

| Audit Artifact / File | Relative File Path | Cryptographic SHA-256 Hash |
| :--- | :--- | :--- |
| **Master Slice Manifest** | `manifests\\slice_manifest.csv` | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |
| **Train Slices CSV** | `splits\\train_slices.csv` | `6b0e2753dbbf780383fdf25f4421f32c496752265d83a348fbda38d2291492cc` |
| **Validation Slices CSV** | `splits\\val_slices.csv` | `4b3d6abd06823239dda131add2eb407d48078fcd17b931acfbff7128425312b1` |
| **Test Slices CSV (Sealed)** | `splits\\test_slices.csv` | `6034134006793c9f5a1392aec51c3e44eb47b54d6ab2e74aca11c8166e1bca0a` |
| **Dataset Version JSON** | `dataset_version.json` | `fb39183db4d59b8010dbb77426e4ba2166a1316c6b2da6925996a41739fd22bf` |
| **Dataset Readiness JSON** | `dataset_readiness.json` | `2dad7f66e37edc57be4e27f1e68707f2d48a5bd432f962ba703734db7bf92bc8` |

---

## 6. Associated Notebooks & Technical Documents

Explore detailed execution scripts and topic documentation:
- 📓 **[01_LiTS_Exploratory_Data_Analysis.ipynb](../notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb)**: Executable EDA notebook.
- 📓 **[02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb](../notebooks/02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb)**: Executable spatial orientation audit.
- 📄 **[08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md](../mark%201%20(part%202)/08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md)**: Full owner provenance record.
- 📄 **[01_PROJECT_AND_DATA_CONTEXT.md](../mark%201%20(part%202)/01_PROJECT_AND_DATA_CONTEXT.md)**: Project context & workspace safeguards.
- 🛠️ **[scripts/arrange_dataset.py](../scripts/arrange_dataset.py)**: Dataset staging build script.

---

[Back to Main README](../README.md) | [Next: Spatial Orientation Forensics](SPATIAL_ORIENTATION_AND_FORENSICS.md)
"""

# 2. docs/SPATIAL_ORIENTATION_AND_FORENSICS.md
doc2 = """# Spatial Orientation Forensics, Denominator Reconciliation & Quality Audit

[![Audit](https://img.shields.io/badge/Audit-Passed-success.svg)]()
[![Containment](https://img.shields.io/badge/ROI%20Containment-100%25-brightgreen.svg)]()

> **Executable Notebook**: [`notebooks/02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb`](../notebooks/02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb)  
> **Source Script Reference**: [`Practice/validation_volume_mask_orientation_forensics.ipynb`](../Practice/validation_volume_mask_orientation_forensics.ipynb)

---

## 1. Spatial Orientation Flips & Matrix Corrections

Auditing revealed that CT slice images and target segmentation masks in Kaggle Part 2 and Hugging Face mirrors suffered from a $180^\\circ$ in-plane rotation flip.

```
       RAW ANOMALOUS ALIGNMENT (47 Volumes)           CORRECTED SPATIAL ALIGNMENT
       ┌───────────────────────────────────┐          ┌───────────────────────────────────┐
       │   Image Matrix    Mask Matrix     │          │   Image Matrix    Mask Matrix     │
       │     (Normal)       (Flipped)      │ ───────► │     (Normal)       (Normal)       │
       │     [  ▲  ]         [  ▼  ]       │ Rot180°  │     [  ▲  ]         [  ▲  ]       │
       └───────────────────────────────────┘          └───────────────────────────────────┘
```

### Planar Rotation Transformation Matrix:

$$\\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} -1 & 0 & 255 \\\\ 0 & -1 & 255 \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}$$

- **`Identity` Matrix (84 Volumes)**: Volumes `0–82` and `100` (Original orientation retained).
- **`Rot180` Matrix (47 Volumes)**: Volumes `83–99` and `101–130` ($180^\\circ$ planar rotation applied).

![Volume Progression](../figures/volume_progression.png)

---

## 2. 4x Tumor Burden Denominator Reconciliation

$$\\text{Legacy Burden } (\\%) = \\frac{\\text{Tumor Pixels}}{512 \\times 512 = 262,144} \\times 100 = 0.0251\\% \\quad (\\text{Deflated } 4\\times)$$

$$\\text{Reconciled Burden } (\\%) = \\frac{\\text{Tumor Pixels}}{256 \\times 256 = 65,536} \\times 100 = 0.1003\\% \\quad (\\text{True Native Burden})$$

* **Impact**: True mean training tumor burden was reconciled from `0.0251%` to **`0.1003%`**.

---

## 3. 2-Stage Bounding Box & ROI Containment Verification

To crop out unneeded background abdominal tissue, a 2-stage ROI extraction protocol was deployed:

1. Predict Stage-1 Liver probability mask at $0.50$ threshold.
2. Select the `largest_3d` connected liver component.
3. Apply $+16\\text{px}$ spatial padding around the bounding box $[y_{\\min}:y_{\\max}, x_{\\min}:x_{\\max}]$.
4. Crop and resize to $256\\times 256$.

![Tumor Bounding Box Analysis](../figures/tumor_bbox_analysis.png)

* **Containment Verification**: Achieved **100% containment** of all tumor pixels across Train and Validation (`152,763 / 152,763` pixels in volume `V116`).

---

## 4. Associated Notebooks & Technical Documents

- 📓 **[02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb](../notebooks/02_Spatial_Orientation_Forensics_and_Quality_Audit.ipynb)**
- 📓 **[Practice/dataset_pairing_forensics.ipynb](../Practice/dataset_pairing_forensics.ipynb)**
- 📄 **[mark 1 (part 2)/01_PROJECT_AND_DATA_CONTEXT.md](../mark%201%20(part%202)/01_PROJECT_AND_DATA_CONTEXT.md)**

---

[Previous: Provenance](DATA_PROVENANCE_AND_ACQUISITION.md) | [Back to Main README](../README.md) | [Next: Radiometrics & HU Windowing](RADIOMETRICS_AND_HU_WINDOWING.md)
"""

# 3. docs/RADIOMETRICS_AND_HU_WINDOWING.md
doc3 = """# Radiometric Hounsfield Unit (HU) Windowing and Intensity Analysis

[![Window](https://img.shields.io/badge/HU%20Window-[-160,%20+240]-blue.svg)]()
[![Modality](https://img.shields.io/badge/Radiometrics-Abdominal%20CT-green.svg)]()

> **Executable Notebook**: [`notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb`](../notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb)  
> **Source Script Reference**: [`Practice/organ_normalized_intensity_robustness_ablation.ipynb`](../Practice/organ_normalized_intensity_robustness_ablation.ipynb)

---

## 1. Principles of Hounsfield Unit (HU) Windowing

Computed Tomography (CT) measurements represent linear attenuation coefficients relative to distilled water ($0\\text{ HU}$) and air ($-1000\\text{ HU}$):

$$\\text{HU} = 1000 \\times \\frac{\\mu_{\\text{tissue}} - \\mu_{\\text{water}}}{\\mu_{\\text{water}} - \\mu_{\\text{air}}}$$

Raw abdominal CT scans capture a wide dynamic range ($-1000\\text{ HU}$ to $+3000\\text{ HU}$). Isolating hepatic lesions requires targeted intensity windowing to suppress uninformative bone and air signals while maximizing soft tissue contrast.

```
       [ -1000 HU ] ----------- [ -160 HU ... +240 HU ] ----------- [ +3000 HU ]
           Air                      Broad Abdominal Window               Bone / Contrast
                                (Parenchyma & Lesion Contrast)
```

---

## 2. Standard Preprocessing Profile: `HU[-160,240]_bilinear_image_nearest_mask_256`

- **Broad Abdominal Window**: `[-160, +240] HU`
- **Min-Max Scaling**: Normalizes intensities linearly to $[0.0, 1.0]$, stored as 8-bit grayscale PNGs ($256\\times 256$).
- **Image Interpolation**: Bilinear
- **Mask Interpolation**: Nearest-Neighbour (preserves integer class labels $0, 1, 2$).

![Preprocessing Pipeline](../figures/preprocessing_pipeline.png)

---

## 3. Normalized Intensity Statistics (8-Bit Scale [0, 255])

Comparing pixel intensity distributions post-windowing demonstrates hyper-intensity of tumor regions relative to dark background tissue.

![Intensity Histograms](../figures/intensity_histograms.png)

| Tissue Region / Parameter | Intensity Mean | Intensity Std | Median HU Value | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **All Image Pixels** | `44.5` | `84.1` | — | Full axial image distribution |
| **Background / Parenchyma** | `44.4` | `84.1` | `+99 HU` (Train) | Abdominal soft tissue baseline |
| **Tumor Lesions** | **`109.1`** | **`102.1`** | `+67 HU` (Train) | Brighter attenuation post-windowing |
| **Tumor-minus-Liver Contrast** | — | — | **`-34 HU`** (Train) | Hypodense parenchymal lesions |
| **Robust Contrast-to-Noise (CNR)** | — | — | **`-1.518`** (Train) | Signal clarity index |

---

## 4. Associated Notebooks & Technical Documents

- 📓 **[01_LiTS_Exploratory_Data_Analysis.ipynb](../notebooks/01_LiTS_Exploratory_Data_Analysis.ipynb)**
- 📓 **[Practice/organ_normalized_intensity_robustness_ablation.ipynb](../Practice/organ_normalized_intensity_robustness_ablation.ipynb)**
- 📄 **[mark 1 (part 2)/step_01_pretraining_dataset_characterization/README.md](../mark%201%20(part%202)/step_01_pretraining_dataset_characterization/README.md)**

---

[Previous: Spatial Forensics](SPATIAL_ORIENTATION_AND_FORENSICS.md) | [Back to Main README](../README.md) | [Next: Patient Splits](PATIENT_SPLITS_AND_BURDEN_ANALYSIS.md)
"""

# 4. docs/PATIENT_SPLITS_AND_BURDEN_ANALYSIS.md
doc4 = """# Patient-Aware Split Partitioning and Tumor Burden Profile

[![Splits](https://img.shields.io/badge/Splits-Patient--Disjoint-blue.svg)]()
[![Partition](https://img.shields.io/badge/Train%2FVal%2FTest-104%2F13%2F14-orange.svg)]()

> **Executable Notebook**: [`notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb`](../notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb)  
> **Source Script Reference**: [`notebooks/split_tumor_audit.ipynb`](../notebooks/split_tumor_audit.ipynb)

---

## 1. Patient-Disjoint Split Composition

To guarantee zero data leakage between adjacent 2D axial slices of the same patient scan, dataset splits are partitioned strictly at the **volume level**.

![Split Comparison](../figures/split_comparison.png)

| Split Name | Volume ID Range | Volume Count | Total Axial Slices | Tumor-Positive Slices | Tumor-Positive Slices % | Mean Volume Tumor Burden % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `0 – 103` | `104` | `40,667` | `4,930` | `12.12%` | `0.1003%` |
| **Validation** | `104 – 116` | `13` | `10,685` | `1,042` | `9.75%` | `0.0874%` |
| **Test (Sealed)** | `117 – 130` | `14` | `7,286` | `1,197` | `16.43%` | `0.3071%` |
| **Total Cohort** | **`0 – 130`** | **131** | **58,638** | **7,169** | **12.23%** | **0.1218%** |

---

## 2. Volume Tumor Burden Profile

Tumor burden represents the percentage of CT volume space occupied by tumor tissue:

$$\\text{Tumor Burden } (\\%) = \\frac{\\sum \\text{Tumor Pixels across Volume}}{\\sum \\text{Volume Pixels}} \\times 100$$

![Tumor Burden per Volume](../figures/tumor_burden_per_volume.png)

### Key Observations:
- **Zero-Tumor Volumes**: `13 / 131` patients ($9.92\\%$) have zero tumors.
- **Low-Burden Volumes ($< 0.05\\%$)**: Combined zero and low-burden volumes represent **84.6% of Train** and **84.6% of Validation**.
- **Test Shift**: The sealed test set contains higher average tumor burden (`0.3071%`), serving as a challenging held-out out-of-distribution evaluation.

![Split Tumor Burden](../figures/split_tumor_burden.png)

---

## 3. Associated Notebooks & Technical Documents

- 📓 **[03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb](../notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb)**
- 📓 **[notebooks/split_tumor_audit.ipynb](../notebooks/split_tumor_audit.ipynb)**
- 📄 **[configs/splits/train_slices.csv](../configs/splits/)**

---

[Previous: Radiometrics](RADIOMETRICS_AND_HU_WINDOWING.md) | [Back to Main README](../README.md) | [Next: Lesion Morphology](LESION_MORPHOLOGY_AND_3D_QUARTILES.md)
"""

# 5. docs/LESION_MORPHOLOGY_AND_3D_QUARTILES.md
doc5 = """# 3D Lesion Morphology and Stratification Bins (Q1 – Q4)

[![Lesions](https://img.shields.io/badge/3D%20Lesions-845%20Components-purple.svg)]()
[![Quartiles](https://img.shields.io/badge/Strata-Q1--Q4-green.svg)]()

> **Executable Notebook**: [`notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb`](../notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb)  
> **Source Script Reference**: [`mark 1 (part 2)/step_01_pretraining_dataset_characterization/step_01_pretraining_dataset_characterization.ipynb`](../mark%201%20(part%202)/step_01_pretraining_dataset_characterization/)

---

## 1. 3D Connected Component Extraction

Across the 131 CT volumes, **845 3D connected lesion components** were extracted using 6-connectivity:

$$V_{\\text{physical}} = N_{\\text{voxels}} \\times \\Delta x \\times \\Delta y \\times \\Delta z \\quad (\\text{mm}^3)$$

$$d_e = 2 \\times \\left( \\frac{3 \\times V_{\\text{physical}}}{4 \\pi} \\right)^{1/3} \\quad (\\text{Equivalent Spherical Diameter in mm})$$

![Tumor Size Distribution](../figures/tumor_size_distribution.png)

---

## 2. Train-Derived Lesion Stratification Bins (`train_derived_lesion_bins.json`)

To prevent validation data leakage, lesion size quartiles were fit strictly on the 637 training set lesions:

| Lesion Size Quartile | Volume Range ($V_L$) | Spherical Diameter ($d_e$) | Clinical Complexity & Detection Requirements |
| :--- | :---: | :---: | :--- |
| **Q1 (Very Small / Small)** | $(-\\infty, 0.173\\text{ mL}]$ | $(-\\infty, 6.92\\text{ mm}]$ | High-resolution feature retention; easily missed by standard downsampling |
| **Q2 (Medium-Small)** | $(0.173, 0.673\\text{ mL}]$ | $(6.92, 10.87\\text{ mm}]$ | Subtle parenchymal lesions |
| **Q3 (Medium-Large)** | $(0.673, 3.944\\text{ mL}]$ | $(10.87, 19.60\\text{ mm}]$ | Moderately defined focal lesions |
| **Q4 (Massive / Large)** | $(3.944, +\\infty\\text{ mL})$ | $(19.60, +\\infty\\text{ mm})$ | Extensive structural deformation (up to $266.35\\text{ mL}$) |

---

## 3. Spatial Occupancy & Heatmap Analysis

Lesions exhibit preferential localization within the central hepatic parenchyma rather than peripheral borders.

![Tumor Heatmap](../figures/tumor_heatmap.png)

---

## 4. Associated Notebooks & Technical Documents

- 📓 **[03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb](../notebooks/03_Patient_Aware_Splits_and_Pretraining_Characterization.ipynb)**
- 📄 **[mark 1 (part 2)/03_PRETRAINING_DATASET_AUDIT_CONTRACT.md](../mark%201%20(part%202)/03_PRETRAINING_DATASET_AUDIT_CONTRACT.md)**

---

[Previous: Patient Splits](PATIENT_SPLITS_AND_BURDEN_ANALYSIS.md) | [Back to Main README](../README.md) | [Next: Model Benchmarks](MODEL_BENCHMARKS_AND_FUSION_POLICY.md)
"""

# 6. docs/MODEL_BENCHMARKS_AND_FUSION_POLICY.md
doc6 = """# Model Architecture, Iteration Benchmarks & Checkpoint Fusion Policy

[![Benchmark](https://img.shields.io/badge/Mark%204E-PASS-success.svg)]()
[![Policy](https://img.shields.io/badge/Fusion-Max%20Probability-blue.svg)]()

> **Executable Notebook**: [`notebooks/04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb`](../notebooks/04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb)  
> **Phase Notebook Reference**: [`mark 1/mark_4e_checkpoint_fusion_validation.ipynb`](../mark%201/mark_4e_checkpoint_fusion_validation.ipynb)

---

## 1. Multi-Stage Experimental Progression (Mark 1 → Mark 4E)

```
Mark 1: Baseline Diagnostics ──► Mark 2: ROI Feasibility ──► Mark 3: 2-Stage Overfit Proof
                                                                         │
Mark 4E: Checkpoint Fusion ◄── Mark 4C/4D: Loss Ablation ◄── Mark 4: Validation Smoke
```

| Phase | Notebook Target | Core Finding / Milestone Result |
| :--- | :--- | :--- |
| **Mark 1** | Baseline probability diagnostic | Identified hypodense lesion suppression on small tumor components |
| **Mark 2** | Multi-window ROI crop feasibility | Confirmed Stage-1 liver crop bounding box reduction ratio ($0.42$) |
| **Mark 3** | 2-Stage overfit verification gate | Achieved hard micro-Dice **`0.900557`** (17 epochs, 100% containment) |
| **Mark 4 / 4B**| 5-Epoch validation smoke training | Revealed high positive predicted-empty rate (`36.85%`) |
| **Mark 4C / 4D**| Loss function ablation & patient reconciliation | Isolated `V116` hypodense lesion localization response failure |
| **Mark 4E** | Controlling Checkpoint Fusion Policy | **PASSED ALL 6 VALIDATION TARGETS** at threshold $0.70$ |

---

## 2. Controlling Checkpoint Fusion Policy (Mark 4E)

To overcome localization failures on low-contrast hypodense lesions (`V104` and `V116`), predictions from a **Control Model** and a **Recall-Loss Model** were combined:

$$P_{\\text{fused}}(x, y) = \\max\\left(P_{\\text{control}}(x, y), P_{\\text{recall\\_loss}}(x, y)\\right)$$

![Sample Overlays](../figures/sample_overlays.png)

---

## 3. Controlling Validation Gate Results (9 Tumor-Positive Validation Patients)

| Benchmark Metric | Target Requirement | Mark 4E Actual Result | Status Gate |
| :--- | :---: | :---: | :---: |
| **Mean Positive-Patient Dice** | $\\ge 0.3329$ | **`0.377087`** | **PASS** |
| **V104 Patient Dice** | $\\ge 0.0500$ | **`0.116627`** | **PASS** |
| **V116 Patient Dice** | $\\ge 0.0100$ | **`0.010474`** | **PASS** |
| **Q1 Small Lesion Detection Rate** | $\\ge 35.0\\%$ | **`50.57%`** | **PASS** |
| **Positive Predicted-Empty Rate** | $\\le 35.0\\%$ | **`27.45%`** | **PASS** |
| **Empty-Slice False Positive Rate** | $\\le 20.0\\%$ | **`5.55%`** | **PASS** |

![Augmentation Examples](../figures/augmentation_examples.png)

---

## 4. Associated Notebooks & Technical Documents

- 📓 **[04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb](../notebooks/04_Model_Benchmarks_and_Checkpoint_Fusion.ipynb)**
- 📓 **[mark 1/mark_4e_checkpoint_fusion_validation.ipynb](../mark%201/mark_4e_checkpoint_fusion_validation.ipynb)**
- 📄 **[mark 1 (part 2)/02_COMPLETED_WORK_AND_VERIFIED_RESULTS.md](../mark%201%20(part%202)/02_COMPLETED_WORK_AND_VERIFIED_RESULTS.md)**

---

[Previous: Lesion Morphology](LESION_MORPHOLOGY_AND_3D_QUARTILES.md) | [Back to Main README](../README.md) | [Next: External Validation](EXTERNAL_VAL_3D_IRCADB.md)
"""

# 7. docs/EXTERNAL_VAL_3D_IRCADB.md
doc7 = """# External Evaluation Protocol: 3D IRCADb-01 Benchmark

[![External Val](https://img.shields.io/badge/External%20Val-3D%20IRCADb--01-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Frozen%20Contract-green.svg)]()

> **Phase Notebook Reference**: [`mark 1 (part 2)/step_13_3d_ircadb_ingestion_and_qc/step_13.ipynb`](../mark%201%20(part%202)/step_13_3d_ircadb_ingestion_and_qc/)  
> **Phase Handoff Contract**: [`mark 1 (part 2)/step_15_frozen_external_evaluation_contract/README.md`](../mark%201%20(part%202)/step_15_frozen_external_evaluation_contract/README.md)

---

## 1. Protocol Overview

To evaluate out-of-domain cross-center generalization without risking data leakage on the sealed LiTS test set, the **3D IRCADb-01** public benchmark dataset was integrated (`step_12` through `step_17` in `mark 1 (part 2)`).

```
        Raw DICOM (IRCAD) ──► NIfTI Conversion ──► HU [-160, +240] ──► 256x256 Parity Build
```

- **Origin**: IRCAD (Research Institute against Digestive Cancer), France.
- **Volume Count**: **20 anonymized abdominal CT scans** ($2,827$ axial slices).
- **Lesion Annotations**: 3D manual segmentations of liver parenchyma, hepatic tumors, portal veins, and adjacent organs.

---

## 2. Parity & Standardization Pipeline

To maintain parity with the LiTS training pipeline, IRCAD scans are converted through identical preprocessing:
1. Resampled in-plane to $256\\times 256$.
2. Standardized using broad abdominal window `[-160, +240] HU`.
3. Evaluated un-tuned under frozen Mark 4E Checkpoint Fusion policy.

---

## 3. Associated Notebooks & Technical Documents

- 📓 **[mark 1 (part 2)/step_13_3d_ircadb_ingestion_and_qc/step_13.ipynb](../mark%201%20(part%202)/step_13_3d_ircadb_ingestion_and_qc/)**
- 📄 **[mark 1 (part 2)/step_12_public_external_dataset_audit_and_acquisition_plan/README.md](../mark%201%20(part%202)/step_12_public_external_dataset_audit_and_acquisition_plan/README.md)**
- 📄 **[mark 1 (part 2)/step_17_external_evaluation_evidence_validation_and_data_card/README.md](../mark%201%20(part%202)/step_17_external_evaluation_evidence_validation_and_data_card/README.md)**

---

[Previous: Model Benchmarks](MODEL_BENCHMARKS_AND_FUSION_POLICY.md) | [Back to Main README](../README.md)
"""

docs_files = {
    "DATA_PROVENANCE_AND_ACQUISITION.md": doc1,
    "SPATIAL_ORIENTATION_AND_FORENSICS.md": doc2,
    "RADIOMETRICS_AND_HU_WINDOWING.md": doc3,
    "PATIENT_SPLITS_AND_BURDEN_ANALYSIS.md": doc4,
    "LESION_MORPHOLOGY_AND_3D_QUARTILES.md": doc5,
    "MODEL_BENCHMARKS_AND_FUSION_POLICY.md": doc6,
    "EXTERNAL_VAL_3D_IRCADB.md": doc7
}

for name, content in docs_files.items():
    filepath = os.path.join("docs", name)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated doc: {filepath}")

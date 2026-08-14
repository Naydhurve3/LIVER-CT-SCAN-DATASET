# Liver CT Tumor Segmentation — Consolidated Evaluation Data

> **Purpose**: Single canonical reference consolidating all data required for evaluating the LiTS-17 liver tumor segmentation project.
> **Sources analyzed**: `README.md`, `dataset.md`, `LITS_DATASET_EDA_GITHUB_CARD.md`, `docs/` (overview, data reference, model benchmarks, external validation), `mark 1/` (Mark 1–4E), `mark 1 (part 2)/` (Steps 01–21, PROJECT_STATUS, final packages), `results/evaluation_report.md`, `understanding the project/`.
> **Generated**: 9 August 2026

---

## 1. Project Identity & Provenance

| Field | Value |
| :--- | :--- |
| Project | Liver CT Scan Dataset & Quality Assurance Platform (LiTS-17) |
| Task | Binary liver tumor/lesion segmentation from 2D axial CT slices |
| Dataset | Liver Tumor Segmentation Benchmark (LiTS-17), 3D Abdominal CT w/ liver & tumor annotations |
| Challenge | LiTS ISBI 2017 / MICCAI 2017–2018, CodaLab Competition ID `17094` |
| Benchmark Publication | Bilic P, et al. *The Liver Tumor Segmentation Benchmark (LiTS).* Medical Image Analysis 2023;84:102680. doi:10.1016/j.media.2022.102680 |
| Repository | https://github.com/Naydhurve3/LIVER-CT-SCAN-DATASET |
| **Canonical Build ID** | `build_corrected_20260713_214847_v2` |
| **Manifest SHA-256** | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |
| Build Directory | `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2` |

### Source Acquisition Record

| Source Mirror | Volume ID Range | Volume Count | Acquired Format |
| :--- | :---: | :---: | :--- |
| Kaggle `andrewmvd/liver-tumor-segmentation` | `0–50` | 51 | NIfTI (`.nii`) |
| Kaggle `...-part-2` | `51–69`, `100` | 20 | NIfTI (`.nii`) |
| Hugging Face CADS `0004_lits` | `70–99`, `101–130` | 60 | `.nii.gz` (decompressed to `.nii`) |
| **Total** | **`0–130`** | **131** | **131 volumes + 131 multiclass segmentations** |

- **Voxel annotation values**: `0 = Background`, `1 = Liver Parenchyma`, `2 = Tumour/Lesion`
- **Modelling target**: Tumor-only binary foreground ($1 = \text{Tumor}, 0 = \text{Background}$)
- **Note**: The internal `test` split is a sealed 14-volume holdout from the 131 annotated training volumes — it is **NOT** the official 70-volume unlabelled LiTS challenge test set.

---

## 2. Cohort Statistics (Global)

| Parameter / Metric | Value |
| :--- | :---: |
| Total patient volumes | `131` (IDs `0`–`130`) |
| Total axial slices | `58,638` |
| Slices per volume | `447.6 ± 274.2` (Range `[74, 987]`, Median `432`) |
| Native resolution | `512 x 512` (standardized → `256 x 256`) |
| Tumor-positive slices | `7,169` (`12.23%`) |
| Organ-positive slices | `19,156` (`32.67%`) |
| Zero-tumor volumes | `13 / 131` (`9.92%`) |
| Pixel imbalance ratio (BG:FG) | `822 : 1` (≈`0.12%` tumor pixels) |
| Overall mean tumor burden | `0.12%` (Max `2.12%`) |
| Organ pixel count | `87,806,713` |
| Tumor pixel count | `4,615,891` |
| Data integrity rate | `100%` (0 corrupt files across `175,914` image/mask PNGs) |

### Spatial Forensics (Corrected Anomalies)

| Anomaly | Fix |
| :--- | :--- |
| 47 volume $180^\circ$ in-plane flips | Rotated Volumes `83–99`, `101–130`; Volumes `0–82`, `100` identity |
| 4× tumor-burden denominator error | Recomputed on `256 x 256` native masks: train burden `0.0251%` → `0.1003%` |
| 30,105 mask values > 255 | Applied `instance mod 256` matching all rows |
| Tumor-positive slice count mismatch | Verified true count `7,169` (legacy listed 7,114) |

---

## 3. Patient-Aware Split Composition

Splits partitioned strictly at the patient/volume level to prevent leakage.

### 3.1 Canonical Split (Corrected Build / `dataset.md`)

| Split | Volume IDs | Volumes | Total Slices | Organ-Positive Slices | Tumor-Positive Slices | Tumor-Positive % | Mean Tumor Burden % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | `0–103` | 104 | 40,667 | 13,382 | 4,930 | 12.12% | 0.1003% |
| **Validation** | `104–116` | 13 | 10,685 | 3,612 | 1,042 | 9.75% | 0.0874% |
| **Test (Sealed)** | `117–130` | 14 | 7,286 | 2,162 | 1,197 | 16.43% | 0.3071% |
| **Total** | `0–130` | **131** | **58,638** | **19,156** | **7,169** | **12.23%** | **0.1218%** |

- Tumor-positive patients: 96 in Train, 9 in Validation (the 9 define Mark 4E validation target cohort), 13 in Test.
- Test holdout is **strictly sealed** — never used for EDA, threshold search, checkpoint selection, or loss tuning.

### 3.2 Legacy EDA Split (docs/PROJECT_OVERVIEW.md, historical)

| Split | Volumes | Slices | Mean slices/vol |
| :--- | :---: | :---: | :---: |
| Train | 104 | 40,667 | 391.0 |
| Val | 13 | 10,685 | 821.9 |
| Test | 14 | 7,286 | 520.4 |

Outlier volumes (28): few slices `< 100` → [0,31,45,54,66,71,72,75,77]; many slices `> 800` → [4,17,18,27,83,87,88,92,94,95,105,108,110,113,114,115,116,117,127].

---

## 4. Geometry & Physical Voxel Dimensions (Train+Val 117 scans)

| Feature Parameter | Train (104 vols) Median | Validation (13 vols) Median | Cohort Range |
| :--- | :---: | :---: | :---: |
| Slices per volume ($N_z$) | 263 | 816 | `[74, 987]` |
| In-plane spacing ($\Delta x,\Delta y$) | 0.772 mm | 0.781 mm | `[0.557, 1.000] mm` |
| Slice thickness ($\Delta z$) | 1.000 mm | 0.800 mm | `[0.450, 5.000] mm` |
| FOV X/Y | 395.5 mm | 399.9 mm | `[285.2, 512.0] mm` |
| FOV Z | 469.5 mm | 635.6 mm | `[74.0, 808.0] mm` |
| Physical liver volume | 1,636 mL | 1,575 mL | `[583, 3,341] mL` |
| Physical tumor volume | 14.3 mL | 3.6 mL | `[0.0, 968.6] mL` |
| ROI crop area ratio | 0.420 | 0.427 | `[0.220, 0.606]` |
| ROI crop height / width | 173 px / 156 px | — | `[114, 235] px` / `[117, 207] px` |

### Frozen 2-Stage ROI Protocol
1. Liver score threshold **0.50** (Stage-1 Liver Net)
2. Select **largest_3D** connected volume component
3. `+16` px padding in X/Y around bounding box
4. Rescale ROI to `256 x 256`
5. **Coverage verified**: 100% tumor-pixel containment (`152,763 / 152,763` px in V116)

---

## 5. Radiometric Profile (HU)

Preprocessing profile: `HU[-160,240]_bilinear_image_nearest_mask_256` with mask op order `derive_resize_nearest_then_transform_256`.

### Normalized Intensity Statistics (0–255)

| Measure | Value |
| :--- | :--- |
| All pixels mean / std | `44.5 / 84.1` |
| Background mean / std | `44.4 / 84.1` |
| Tumor tissue mean / std | `109.1 / 102.1` |
| Median tumor-minus-liver contrast | `-34 HU` (Train), `-44 HU` (Validation) |
| Robust Contrast-to-Noise Ratio (CNR) | `-1.518` (Train), `-1.881` (Validation) |

### Raw HU Stats (Train vs Validation)

| Measure | Train | Validation |
| :--- | :---: | :---: |
| Tumor HU median | +67 HU | +59 HU |
| Tumor HU IQR | 36 HU | 39 HU |
| Liver HU median | +99 HU | +106 HU |
| Tumor-minus-liver contrast | -34 HU | -44 HU |
| Robust CNR | -1.52 | -1.88 |
| Background HU median | -887 HU | -908 HU |

**Windowing**: Broad abdominal window `[-160, +240] HU` mapped to `[0,1]` (8-bit grayscale 256×256 PNGs); imaging stored via bilinear interpolation images + nearest-neighbour masks.

---

## 6. 3D Lesion Morphology & Stratification

Total **845 distinct 3D 6-connected lesion components** (Train 637, Validation 208).

| Lesion Metric | Value |
| :--- | :---: |
| 3D lesion volume (median) | `0.383 mL` (range `[0.00035, 968.60] mL`) |
| Equivalent spherical diameter | `9.01 mm` (range `[0.87, 122.70] mm`) |
| Axial span ($N_z$) | 7 slices median |
| Lesions per patient | 4 median (max 70) |
| Max tumor-to-liver ratio | 0.314 |

### Train-Derived Lesion Stratification Bins (`train_derived_lesion_bins.json`)

| Stratum | Tumor Volume ($V_L$) | Spherical Diameter ($d_e$) |
| :--- | :--- | :--- |
| Q1 (Very small/small) | $(-\infty, 0.173\text{ mL}]$ | $(-\infty, 6.92\text{ mm}]$ |
| Q2 (Medium-small) | $(0.173, 0.673\text{ mL}]$ | $(6.92, 10.87\text{ mm}]$ |
| Q3 (Medium-large) | $(0.673, 3.944\text{ mL}]$ | $(10.87, 19.60\text{ mm}]$ |
| Q4 (Large/massive) | $(3.944, \infty\text{ mL})$ | $(19.60, +\infty\text{ mm})$ |

- Zero-tumor + low-burden (< 0.05%) patients = **84.6% of Train** and **84.6% of Validation**.

---

## 7. Model & Experimental Stage Progression (Mark 1 → Mark 4E)

| Phase | Purpose / Gate | Key Result |
| :--- | :--- | :--- |
| **Mark 1** | Baseline probability-contrast localization diagnostic | Identified suppression of hypodense lesions |
| **Mark 2** | Multi-window & predicted-liver ROI feasibility | ROI crop reduction ratio ~0.42 |
| **Mark 3** | 2-Stage deterministic overfit gate | Hard micro-Dice `0.900557` (17 epochs); 100% containment |
| **Mark 4 / 4B** | 5-epoch validation smoke | High positive predicted-empty `36.85%`; threshold-only repair rejected |
| **Mark 4C / 4D** | Loss ablation / metric reconciliation | Recall-loss improves recall but damages V116 (`0.000712`) |
| **Mark 4E** | Checkpoint fusion validation gate | **PASSED ALL 6 VALIDATION TARGETS** @ `0.70` |

### Mark 4E Fusion Policy (Validation Benchmark)

$$\text{Fused Probability} = \max(P_{\text{control}}, P_{\text{recall\_loss}})$$

| Evaluation Metric | Target | Mark 4E Result | Status |
| :--- | :---: | :---: | :---: |
| Mean Dice (9 Tumor-Positive Val Patients) | ≥ 0.3329 | **0.3771** | ✔ PASS |
| V104 Patient Dice | ≥ 0.0500 | **0.1166** | ✔ PASS |
| V116 Patient Dice | ≥ 0.0100 | **0.0105** | ✔ PASS |
| Q1 Small Lesion Detection Rate | ≥ 35.0% | **50.57%** | ✔ PASS |
| Positive Predicted-Empty Rate | ≤ 35.0% | **27.45%** | ✔ PASS |
| Empty-Slice False Positive Rate | ≤ 20.0% | **5.55%** | ✔ PASS |

### Reference / Benchmark Results (UP³RE-Net research, `results/evaluation_report.md`)

Dataset (for reference table): 131 volumes / 58,638 slices / tumor slices 7,114 (12.13%) / imbalance 1,140.39:1.

| Model | Dice | IoU | HD95 | ASD | NSD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MobileNetV2-UNet | 0.8787 ± 0.3265 | 0.8787 ± 0.3265 | 644.4742 ± 827.1965 | 644.4742 ± 827.1965 | 0.0000 ± 0.0000 |
| UP³RE-M0 | 0.8263 ± 0.3691 | 0.8212 ± 0.3736 | — | — | — |
| UP³RE-M1 | 0.8787 ± 0.3265 | 0.8787 ± 0.3265 | — | — | — |
| UP³RE-M2 | 0.7318 ± 0.4337 | 0.7264 ± 0.4376 | — | — | — |
| UP³RE-Ensemble | 0.8867 ± 0.3091 | 0.8832 ± 0.3133 | 329.7997 ± 622.7196 | 324.1874 ± 625.2756 | 0.1145 ± 0.2282 |

---

## 8. Frozen Inference Policy (Step 03 — Final)

| Parameter | Value |
| :--- | :--- |
| Backbone | MobileNetV2-UNet (control + recall-loss) |
| Input | Single broad CT window `[-160,240]` HU, uint8 ÷ 255 |
| ROI | Predicted-liver: threshold 0.50, largest 26-connectivity component, padding 16, full-image fallback |
| Fusion | Pixelwise maximum of control & recall-loss probabilities |
| Threshold | Global 0.70 (no post-processing) |
| Epsilon / seed | 1e-6 / 42 |
| Q1 slice def | 1–51 tumor pixels (51 = train-only 25th percentile) |
| Checkpoint hashes | Control `9b0c7749af66b0fc3808f6757d0d90af01384e06afe02090361b220df48b6e8b`; Recall `c01eb4b81e4e7f1d84c7966aca56e738d87d06d404907f0bcc7c67a79ed4ec4d` |

---

## 9. LiTS Internal-Holdout Test Results (One-Time, Step 04)

**Run UUID**: `871d289b-bf6b-4346-978f-2df02ade26ab` — sealed, `rerun_allowed: false`

| Metric | Value |
| :--- | :---: |
| Global Dice | **0.767696** |
| Mean positive-patient Dice | **0.507297** (bootstrap 95% CI 0.3442–0.6594) |
| Median positive-patient Dice | 0.5651 |
| Minimum positive-patient Dice | **0.000000** |
| Global pixel precision | 0.844394 |
| Global pixel recall | 0.703771 |
| Q1 detection (train-edge, 1–51 px) | 47.552% |
| Positive predicted-empty | 9.942% |
| Empty-slice false positives | 3.810% |
| Lesion-strata detection | Q1 60.29%, Q2 98%, Q3 97.37%, Q4 100% |

### Formal Decision
`FINAL_TEST_COMPLETE` — **Formal acceptance FAILED**: V121 scored ~0 Dice vs predeclared minimum positive-patient floor `0.01`. All 526 tumor pixels in V121 were ROI-contained → recognition failure, not clipping.
- Decision code: `REPORT_FINAL_TEST_GATE_FAILURE_NO_TUNING_NO_RERUN`.

---

## 10. External Evaluation: 3D-IRCADb-01 (Steps 12–17)

### Cohort
- 20 anonymized abdominal CT scans (`2,823` slices evaluated), 15 tumor-positive, 5 tumor-negative, zero exclusions
- License **CC BY-NC-ND 4.0** (verified 5 Aug 2026), cite Soler et al. (2010), IRCAD
- Ingestion: DICOM → NIfTI, `256×256`, HU `[-160,240]` parity build (24 source lesion-folder proxies)
- External stratum coverage (Step 15): patient Q1=1, Q2=7, Q3=0, Q4=7

### Results (Step 16 authorized, UUID `61d1f140-c8b4-436a-928a-1e4b6f7c0b56`)

| Metric | Value |
| :--- | :---: |
| Global Dice | **0.847977** |
| Pixel precision | 0.823699 |
| Pixel recall | 0.873729 |
| Mean positive-patient Dice | **0.711196** (95% bootstrap CI 0.567803–0.827823) |
| Median positive-patient Dice | 0.835514 |
| Minimum positive-patient Dice | 0.012992 (`ircadb_18`) |
| Q1 positive-slice detection | 50.00% |
| Positive predicted-empty | 9.86% |
| Empty-slice FP | 11.57% |
| Smallest-lesion stratum detection | 63.64% (14/22 components) |
| Burden split | Q1 0.7178, Q2 0.5443 (min 0.0129), Q4 0.8772 |

- All 5 tumor-negative controls produced some predicted pixels (total **52.936 mL** standardized physical FP volume); cases 7 & 14 dominate → false positives under frozen truth.
- Case 18 genuine failure (16 px intersect `livertumor`, recall 1.41%). Case 7: 86.39% of pixels overlap excluded `tumor` mask; Case 14: 88.93% overlap `metastasectomie`.
- Decision: `REPORT_EXTERNAL_GENERALIZATION_PASS_NO_TUNING`; Step 16 signature `4fb3d6d14010a982912c9ca8e85854e326960fa4a60543da47a378c6fd1b13aa`.

---

## 11. Appearance / Success Criteria & Historical Diagnostics

### Multi-item study results (`understanding` repo, older)

| Experiment | Best Epoch | Global Dice | Mean patient Dice | V104 | Q1 det | Positive empty | Empty FP | Decision |
|---|---|---|---|---|---|---|---|---|
| 16-slice overfit | 76 | 0.816306 | — | — | — | — | — | Pass |
| 5-epoch smoke | 5 | 0.461613 | — | — | — | 27.74% | 14.42% | Pass |
| Patient baseline | 9 | 0.558357 | 0.383 | 0.198 | ≈0 | — | — | Revise |
| Lesion sampler | 7 | 0.524200 | 0.356 | 0.152 | 0.0012 | 53.99% | 17.83% | Fail |
| Composite loss | 5 | 0.568154 | 0.323 | 0.005 | 0.051 | 29.66% | 3.03% | Fail |
| Organ-assisted | 8 | 0.546132 | 0.407 | 0.587 | 0.027 | 60.08% | 47.42% | Fail |
| 2.5D context | 10 | 0.553527 | 0.356 | 0.0006 | 0.0009 | 25.48% | 2.37% | Fail |
| 3D post-processing | val | — | 0.406964 | 0.587 | 0.027 | 60.08% | 47.43% | Fail |
| Multi-task liver/tumor | 8 | — | 0.332854 | ≈0 | ≈0 | 28.14% | 3.25% | Fail |

### Appearance-shift diagnostic
- Conv/Sampling threshold P95: `6.4022`; Volume 104 shift `6.6768`; Volume 116 shift `3.7398` (V104 was outlier; V116 not explained by aggregate shift).

---

## 12. Success Criteria / Gates (Formal Acceptance Contract)

Temporary six targets (validation): Mean Dice ≥ 0.3329, V104 ≥ 0.05, V116 ≥ 0.01, Q1 detection ≥ 35%, positive predicted-empty ≤ 35%, empty-slice FP ≤ 20% — all mark4E passed.

Final test acceptance adds: catastrophic positive-patient failure floor (V121 ≥ 0.01), finite/unique/complete evaluation, frozen policy only. **Outcome: 7/8 passed; model rejected on V121.**

Decision vocabulary status levels: `TEMPORARY_CONTINUATION_PASS`, `VALIDATION_FREEZE_PASS`, `FINAL_TEST_COMPLETE`, `FINAL_PROJECT_COMPLETE`, `EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS`, etc.

---

## 13. Integrity & Reproducibility Register

### 13.1 File hashes (SHA-256) — verbatim from `dataset.md` §13

| Artifact | Relative Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Master Slice Manifest** | `manifests\slice_manifest.csv` | `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889` |
| **Train Slices CSV** | `splits\train_slices.csv` | `6b0e2753dbbf780383fdf25f4421f32c496752265d83a348fbda38d2291492cc` |
| **Train Volumes TXT** | `splits\train_volumes.txt` | `6183e84acb5a30fd04a38875f54c6a02b4f64f588725f236ff3984ef4450a62d` |
| **Validation Slices CSV** | `splits\val_slices.csv` | `4b3d6abd06823239dda131add2eb407d48078fcd17b931acfbff7128425312b1` |
| **Validation Volumes TXT** | `splits\val_volumes.txt` | `a28fea675229371093748adb90e9c1d23dd2a5699fd6f833a902ee4681fdb9ad` |
| **Test Slices CSV (Sealed)** | `splits\test_slices.csv` | `6034134006793c9f5a1392aec51c3e44eb47b54d6ab2e74aca11c8166e1bca0a` |
| **Test Volumes TXT (Sealed)** | `splits\test_volumes.txt` | `830f8ded37afeb01347c5f222d91d253ac11cad8bc9621ce8cb04b0ef6b8d154` |
| **Corrected `dataset_version.json`** | `dataset_version.json` | `fb39183db4d59b8010dbb77426e4ba2166a1316c6b2da6925996a41739fd22bf` |
| **Corrected `dataset_readiness.json`** | `dataset_readiness.json` | `2dad7f66e37edc57be4e27f1e68707f2d48a5bd432f962ba703734db7bf92bc8` |
| **Source Registry JSON** | `00_source_registry\source_registry.json` | `4835f1bf7fe1ba0e51dff9f52a2a463590cb6b90fd2f823a53b342d13073f173` |

### 13.2 Controlling / Signed Artifacts (verify against Step 05 package)

| Artifact | Value |
| :--- | :--- |
| Frozen policy config SHA-256 | `52d2df856709254992ab6b59f84a213a0a00f6e67293f66665b6f258d01f2131` |
| Final acceptance contract SHA-256 | `12e8932d71651fd9247d841a86dfcd0fa80d64d74c4f334e29b113c28213b52d` |
| One-time LiTS test run UUID | `871d289b-bf6b-4346-978f-2df02ade26ab` |
| Step 04 evidence inventory SHA-256 | `9b61a09119128c7b85689762aee51617ee791a2a2b1aafc56adf185dbe6672ab` |
| External 3D-IRCADb-01 run UUID | `61d1f140-c8b4-436a-928a-1e4b6f7c0b56` |

> **IMPORTANT**: The full verbatim 10-row hash register must be kept as-is; for
> exact reproduction copy hashes directly from `dataset.md` §13 and the
> `step_05_final_research_package/outputs/REPRODUCIBILITY_INSTRUCTIONS.md`
> provenance (frozen-policy, acceptance-contract, evidence-inventory hashes).
> `dataset_readiness.json` / `dataset_version.json` schemas are described in
> `dataset.md` §9.

## 14. Pipeline Flow Diagram

```mermaid
flowchart LR
    src[Raw NIfTI .nii] --> norm["HU [-160,240] clip → min-max → bilinear → 256x256"]
    norm --> build[build_corrected_20260713_214847_v2]
    build --> splits[Patient-disjoint splits]
    splits --> roi[Stage-1 ROI crop]
    roi --> fuse[max(control,recall) @ 0.70]
    fuse --> test[One-time sealed test]
    fuse --> ext[External 3D-IRCADb-01]
```

---

## References / Citations

- Bilic P, et al. The Liver Tumor Segmentation Benchmark (LiTS). Medical Image Analysis 2023;84:102680. doi:10.1016/j.media.2022.102680
- Soler L, et al. 3D image reconstruction for comparison of algorithm database: a patient specific anatomical and medical image database. IRCAD, Strasbourg, France, Technical Report 2010. (3D-IRCADb-01; no DOI exists)
- own citation: Nayd Hurve, "LiTS-17 CT Dataset Quality Audit...", 2026, GitHub.

---

> **Sources of evidence (primary)**: `dataset.md` (authoritative manifest figures), `README.md`, `LITS_DATASET_EDA_GITHUB_CARD.md`, `docs/MODEL_BENCHMARKS_AND_FUSION_POLICY.md`, `docs/EXTERNAL_VAL_3D_ICCAD.md`, `mark 1 (part 2)/02_COMPLETED_WORK_AND_VERIFIED_RESULTS.md`, `PROJECT_STATUS.md`, `step_04/.../TEST_RESULTS_DATA_CARD.md`, `step_05/.../FINAL_PROJECT_DATA_CARD.md` & `FINAL_TECHNICAL_REPORT.md`, `step_05/.../REPRODUCIBILITY_INSTRUCTIONS.md`, `step_17/.../EXTERNAL_EVALUATION_DATA_CARD.md`, `results/evaluation_report.md`, `understanding the project/05_complete_results.md`.
>
> **Copy/Uses**: Keep `dataset.md`'s precise field names and hash values authoritative for any further computation.
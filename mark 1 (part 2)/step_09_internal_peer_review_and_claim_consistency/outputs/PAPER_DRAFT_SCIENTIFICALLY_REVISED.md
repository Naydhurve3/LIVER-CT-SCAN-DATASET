<!-- Internal scientific consistency review completed. This copy is venue-neutral and still requires owner declarations, authoritative dataset terms and venue selection. -->

# Maximum-Probability Fusion of Two MobileNetV2 U-Nets for Liver-Tumour Segmentation: A Reproducible Held-Out Evaluation

## Abstract

**Background:** Liver-tumour segmentation remains difficult for small and low-contrast lesions. **Methods:** We evaluated a frozen two-stage pipeline on the corrected LiTS build. A predicted-liver ROI preceded two MobileNetV2 U-Net tumour models. Their sigmoid scores were fused by pixelwise maximum and thresholded globally at 0.70. All preprocessing, checkpoints, acceptance targets and metrics were frozen before one-time test access. **Results:** Across 13 tumour-positive test patients, mean patient Dice was 0.5073 (95% patient-bootstrap interval 0.3442–0.6594); global Dice was 0.7677, precision 0.8444, and recall 0.7038. Smallest-quartile lesion detection was 60.3%. One patient had effectively zero Dice despite complete ROI containment, causing failure of the predeclared minimum-patient acceptance gate. **Conclusion:** Maximum-probability fusion achieved strong aggregate overlap on the one-time held-out LiTS test subset but did not provide reliable patient-level performance. The outcome should be reported as a completed negative formal gate rather than deployment-ready success.

**Keywords:** liver tumour, CT, semantic segmentation, LiTS, MobileNetV2 U-Net, model fusion, reproducibility

## 1. Introduction

Automated liver-tumour delineation is studied as a component of quantitative oncology research workflows, but lesion size, contrast, multiplicity and acquisition variability create strong case-level heterogeneity. This study asked whether a validation-selected, maximum-probability fusion of complementary tumour models would reproduce under a strictly frozen one-time held-out evaluation.

## 2. Materials and methods

### 2.1 Dataset governance

The study used the LiTS benchmark dataset [Bilic2023LiTS] and corrected build `build_corrected_20260713_214847_v2`. Patient-disjoint training, validation and test splits were controlled by a SHA-256-locked manifest. Dataset integrity, geometry, label semantics, train-validation leakage, morphology, HU contrast, difficulty and focus-case phenotypes were audited before the final model freeze.

### 2.2 Preprocessing and ROI generation

Derived axial images represented a broad `[-160,240]` HU window at 256×256 resolution. A frozen two-output MobileNetV2 U-Net generated liver scores. The largest 26-connected 3D liver component at threshold 0.50 was projected to a bounding box padded by 16 pixels; crops were resized to 256×256 and mapped back using bilinear score interpolation.

### 2.3 Tumour models and fusion

Control and recall-loss U-Net-family [Ronneberger2015UNet] checkpoints with MobileNetV2 encoders [Sandler2018MobileNetV2] produced sigmoid tumour scores. Fusion was `max(p_control,p_recall)`. A single global threshold of 0.70 and no post-processing were used for every patient.

### 2.4 Outcomes and uncertainty

The primary aggregate outcome was mean Dice over tumour-positive patients. Secondary outcomes included global Dice, pixel precision/recall, per-patient Dice, train-edge Q1 slice detection, positive predicted-empty rate, empty-slice false-positive rate, and matched 3D lesion performance by train-derived physical-volume quartiles. Uncertainty used 10,000 patient-bootstrap resamples with seed 42.

### 2.5 Governance and test lock

Following a transparent medical-imaging-AI reporting approach [Tejani2024CLAIM], the inference policy and formal acceptance table were frozen before test authorization. Test data were evaluated once under run UUID `871d289b-bf6b-4346-978f-2df02ade26ab`. The completed ledger prohibits rerun and test-driven tuning.

## 3. Results

The test cohort contained 7,286 slices from 14 volumes; 13 volumes were tumour-positive. Global Dice was 0.7677. Mean positive-patient Dice was 0.5073, with median 0.5651 and 95% bootstrap interval 0.3442–0.6594. Global precision and recall were 0.8444 and 0.7038. Q1 slice detection was 47.6%, positive predicted-empty 9.9%, and empty-slice false positives 3.8%.

Seven of eight formal test/integrity targets passed. The minimum positive-patient Dice target failed: V121 had effectively zero Dice versus the 0.01 floor. V121 tumour pixels were fully contained by the ROI, but cached control and recall scores did not reach the global threshold in the truth region.

Smallest-quartile lesion detection was 60.3% with mean matched Dice 0.238; larger quartiles achieved at least 97.4% detection.

## 4. Discussion

The frozen fusion policy produced strong aggregate overlap on the one-time held-out evaluation and met the aggregate cohort targets while failing the patient-level guardrail. However, the complete V121 miss and several other low-Dice patients show that aggregate overlap is insufficient for reliability claims. The smallest lesion stratum remains the clearest systematic weakness. Because the test was used once under a predeclared contract, these findings are descriptive failure evidence, not permission for retrospective selection.

## 5. Limitations

The held-out cohort is small; the pipeline is two-dimensional and based on derived broad-window images; probabilities are not clinically calibrated; and no external institutional cohort or prospective evaluation was available. Patient-level failure mechanisms require a future separately governed study.

## 6. Conclusion

Maximum-probability fusion provided strong aggregate LiTS performance but failed the predeclared catastrophic-patient guardrail. The technically correct conclusion is a completed held-out evaluation with failed formal model acceptance.

## Declarations and references to complete before submission

Add dataset licensing, ethics/applicability statement, author contributions, conflicts, funding, code availability, and literature references during manuscript preparation. No external citations were fabricated in this internal draft.


## Candidate references (verify venue style before submission)

- [Bilic2023LiTS] Bilic P, Christ PF, Li HB, et al.. The Liver Tumor Segmentation Benchmark (LiTS). Medical Image Analysis 84:102680. 2023. doi:10.1016/j.media.2022.102680
- [Ronneberger2015UNet] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. 2015. doi:10.1007/978-3-319-24574-4_28
- [Sandler2018MobileNetV2] Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018. 2018. doi:10.1109/CVPR.2018.00474
- [Tejani2024CLAIM] Tejani AS, Klontzas ME, Gatti AA, et al.. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. Radiology: Artificial Intelligence 6(4):e240300. 2024. doi:10.1148/ryai.240300

## Submission declarations

[OWNER REQUIRED: complete `DECLARATION_TEMPLATE.md`; unresolved placeholders prohibit submission.]

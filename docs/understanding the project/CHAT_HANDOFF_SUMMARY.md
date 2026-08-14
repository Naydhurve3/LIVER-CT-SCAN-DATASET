# Liver Tumor Segmentation — Chat Handoff Summary

Use this document to brief another AI assistant, researcher, or collaborator. The goal is to obtain guidance on the next scientifically justified procedure without repeating completed experiments or accessing the locked test split.

## 1. Project objective

The project is developing a reproducible liver-tumor segmentation pipeline using the LiTS CT dataset. The immediate research objective is not simply to maximize aggregate Dice. It is to obtain a model that generalizes across validation patients, detects small lesions, avoids empty predictions on tumor-positive slices, and controls false positives on tumor-negative slices.

The work follows gated validation:

1. Verify dataset pairing, orientation, masks, and splits.
2. Prove that the loader/model can overfit a deterministic subset.
3. Run a short baseline.
4. Evaluate patient-level and lesion-size failures.
5. Run one controlled ablation at a time.
6. Keep the test split locked until all validation targets pass.

## 2. Authoritative workspace

- Project root: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`
- Main evidence and notebook directory: `Practice\`
- Documentation directory: `understanding the project\`
- Dataset build:
  `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- Manifest:
  `manifests\slice_manifest.csv`
- Manifest SHA-256:
  `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`

`Practice/` is the authoritative project evidence store. Older notebooks and builds must not override the corrected manifest-driven pipeline.

## 3. Dataset

### Size and split

- Total slices: `58,638`
- Total patients/volumes: `131`
- Stored size: `256 × 256`
- Manifest rows are unique by `sample_id`.
- All rows are `verified`.
- All volumes are manually spatially `approved`.

| Split | Slices | Volumes | Tumor-positive slices |
|---|---:|---:|---:|
| Train | 40,667 | 104 | 4,930 |
| Validation | 10,685 | 13 | 1,042 |
| Test | 7,286 | 14 | 1,197 |

The train, validation, and test volume sets are patient-disjoint.

### Files per slice

Each manifest row contains:

- CT image PNG.
- Binary liver/organ mask PNG.
- Binary tumor mask PNG.
- Source volume and segmentation NIfTI paths.
- Source SHA-256 hashes.
- Volume ID and axial slice index.
- Orientation transform.
- Organ and tumor pixel counts.
- Split, integrity, verification, preprocessing, and spatial-approval fields.

### Preprocessing recorded by the corrected build

- Source window profile: HU `[-160, 240]`.
- Image resize: bilinear.
- Mask resize: nearest neighbor.
- Output size: `256 × 256`.
- Model target: tumor-only foreground.
- Organ mask is stored separately.
- Tumor containment inside the organ mask was verified.

The PNGs are already derived/windowed. Do not apply a second HU window to the PNGs unless creating a newly versioned build from source NIfTI.

## 4. Dataset problems already found and fixed

The initial data pipeline contained in-plane orientation errors.

- All 156 sampled source-to-derived regenerations matched their named source slice.
- Slice-offset search selected offset zero.
- The issue was not slice numbering.
- Volumes `83–99` and `101–130` required 180-degree correction.
- Volumes `0–82` and `100` used identity orientation.
- There were 47 critical orientation failures in the earlier build.
- The corrected v2 build is the only build used for the recorded modeling experiments.

An earlier tumor-burden calculation also used a hard-coded `512×512` denominator for `256×256` masks, under-reporting tumor burden by four times. This was corrected to use native mask dimensions.

## 5. Loader and overfit verification

Notebook:

`Practice\verified_manifest_loader_and_overfit_gate.ipynb`

Checks included:

- Manifest hash and schema.
- Unique IDs and path validation.
- Patient-disjoint splits.
- Test-loader lock.
- Orientation and binary masks.
- Manifest pixel counts versus loaded masks.
- Deterministic 16-slice training subset.
- MobileNetV2-U-Net overfit.
- Safe interruption, resume, and thermal monitoring.

Result:

- Loader gate passed.
- Overfit gate passed.
- Selected slices: `16`.
- Completed epochs: `76`.
- Required hard micro-Dice: `0.80`.
- Achieved hard micro-Dice: `0.816306`.
- Threshold: `0.50`.
- Device: CUDA with mixed precision.
- Pretrained encoder: false.
- Test images accessed: false.

This proves the corrected pipeline can fit a small deterministic subset. It does not prove generalization.

## 6. Metrics used for every serious decision

Aggregate/global Dice is not accepted alone because large lesions and strong patients can dominate it.

Required diagnostics:

- Global pixel micro-Dice.
- Mean, median, and worst positive-patient Dice.
- Per-volume Dice.
- Volume 104 and volume 116 Dice.
- Pixel precision and recall.
- Positive-slice recall.
- Percentage of positive slices predicted completely empty.
- Empty-slice false-positive percentage.
- Q1–Q4 lesion-size results.
- Threshold sensitivity.
- Expected-versus-actual targets.

The current multi-task targets are:

| Metric | Target |
|---|---:|
| Mean positive-patient Dice | `≥0.406915` |
| Volume 104 Dice | `≥0.50` |
| Volume 116 Dice | `≥0.05` |
| Q1 smallest-lesion detection | `≥45%` |
| Positive predicted-empty | `≤20%` |
| Empty-slice false positives | `≤15%` |
| Mean liver Dice | `≥0.90` |

## 7. Experiment history and results

### Five-epoch manifest baseline smoke test

Model and training:

- MobileNetV2-U-Net.
- One image channel and one tumor output.
- Focal-Dice loss.
- Batch size 8; validation batch 16.
- Learning rate `3e-4`.
- Weight decay `1e-4`.
- Positive-slice sampling weight 4.
- Threshold `0.50`.
- No pretrained encoder.

Result:

- Gate passed.
- Best epoch: 5.
- Global validation micro-Dice: `0.461613`.
- Positive-slice recall: `56.05%`.
- Positive predicted-empty: `27.74%`.
- Empty-slice false positives: `14.42%`.

### Patient-aware baseline to epoch 10

Configuration:

- Focal-Dice.
- Positive-slice sampling weight 3.
- Learning rate `3e-4`.
- AdamW and cosine schedule.

Result:

- Best epoch: 9.
- Global micro-Dice: `0.558357`.
- Mean patient Dice: `0.383069`.
- Median patient Dice: `0.342593`.
- Worst patient Dice: approximately zero.
- Volume 104 Dice: `0.197613`.
- Volume 116 Dice: approximately zero.

Conclusion: aggregate Dice looked reasonable, but important patients failed.

### Patient/lesion-balanced sampler

Technique:

- Inverse-volume × lesion-size-quartile sampling.

Result:

- Best epoch: 7.
- Global Dice: `0.524200`.
- Mean patient Dice: `0.356189`.
- Volume 104: `0.152444`.
- Volume 116: `0.001856`.
- Q1 detection: `53.99%`.
- Positive predicted-empty: `19.29%`.
- Empty-slice false positives: `17.83%`.

Conclusion: small-lesion detection improved, but overall patient performance worsened.

### Pure recall-aware Focal-Tversky

Parameters:

- Alpha `0.40`.
- Beta `0.60`.
- Gamma `0.75`.

Result:

- Collapsed to empty tumor prediction.
- Mean patient Dice approximately zero.
- Q1 detection: `0%`.
- Positive predicted-empty: `100%`.
- Empty-slice false positives: `0%`.

Conclusion: do not repeat this configuration.

### Stabilized composite loss

Technique:

- `0.75` Focal-Dice plus `0.25` Focal-Tversky.

Result:

- Best epoch: 5.
- Global Dice: `0.568154`.
- Mean patient Dice: `0.322939`.
- Volume 104: `0.005005`.
- Volume 116: `0.051481`.
- Q1 detection: `29.66%`.
- Positive predicted-empty: `41.46%`.
- Empty-slice false positives: `3.03%`.

Conclusion: it recovered volume 116 slightly but harmed the broader patient cohort and recall.

### Appearance/domain forensics

Results:

- Sampled slices: `3,106`.
- Training P95 appearance-shift threshold: `6.4022`.
- Volume 104 shift score: `6.6768`, flagged as an appearance outlier.
- Volume 116 shift score: `3.7398`, below that threshold.

Conclusion: volume 104 has a clear appearance-domain issue. Volume 116 is not explained by the same aggregate shift score.

### Organ-assisted intensity normalization

Technique:

- Median and IQR statistics calculated within the ground-truth organ mask.
- Robust z-score clipped to ±3 and rescaled.
- Gamma range `0.85–1.15`.
- Gaussian noise standard deviation up to `0.025`.
- Focal-Dice and positive-slice weight 3.

Result:

- Best epoch: 8.
- Global Dice: `0.546132`.
- Mean patient Dice: `0.406915`.
- Volume 104 Dice: `0.586500`.
- Volume 116 Dice: `0.026782`.
- Q1 detection: `60.08%`.
- Positive predicted-empty: `8.25%`.
- Empty-slice false positives: `47.42%`.

Interpretation:

- This is the strongest observed mean patient and volume 104 result.
- It failed volume 116 and false-positive guardrails.
- It is diagnostic, not deployable, because ground-truth organ masks were used to compute validation normalization statistics.

### Adjacent-slice 2.5D context

Technique:

- Three input channels: previous, current, and next axial slices.

Result:

- Best epoch: 10.
- Global Dice: `0.553527`.
- Mean patient Dice: `0.355938`.
- Volume 104: `0.000636`.
- Volume 116: `0.000932`.
- Q1 detection: `25.48%`.
- Positive predicted-empty: `42.99%`.
- Empty-slice false positives: `2.37%`.

Conclusion: 2.5D context suppressed false positives but also suppressed true lesions.

### Validation-only 3D post-processing

Techniques:

- Threshold sweep.
- 3D connected components.
- Hysteresis thresholds.
- Minimum component size.
- Minimum axial span.

Best configuration:

- Unchanged raw threshold `0.50`.

Result:

- Mean patient Dice: `0.406964`.
- Volume 104: `0.586555`.
- Volume 116: `0.026794`.
- Q1 detection: `60.08%`.
- Positive predicted-empty: `8.25%`.
- Empty-slice false positives: `47.43%`.

Conclusion: post-processing could not remove false positives without losing patient Dice.

### Multi-task liver and tumor localization

Technique:

- One shared MobileNetV2-U-Net.
- Output channel 0: liver.
- Output channel 1: tumor.
- Image-only robust normalization; no ground-truth organ used for validation preprocessing.
- Liver masks used as training supervision.
- Tumor predictions gated by predicted-liver support.
- Warm start from the organ-assisted intensity checkpoint.
- Tumor loss weight `0.72`.
- Liver loss weight `0.23`.
- Containment loss weight `0.05`.
- Learning rate `1e-4`.
- Positive-slice weight 3.
- Tumor/liver thresholds `0.50`.
- Liver dilation kernel `31`.

Continuation:

- Completed through epoch 10.
- Thermal waiting and exact resume worked.
- Best epoch: 8.

Best result:

- Mean patient Dice: `0.332854`.
- Mean liver Dice: `0.886322`.
- Volume 104 Dice: approximately zero.
- Volume 116 Dice: approximately zero.
- Q1 detection: `28.14%`.
- Positive predicted-empty: `46.16%`.
- Empty-slice false positives: `3.25%`.
- Targets passed: 1 of 7.

Important diagnostic:

- Raw tumor and liver-gated tumor predictions were exactly identical.
- The predicted-liver gate removed zero pixels.
- Therefore, remaining mistakes are inside the predicted-liver support.
- The model reduced false positives by strongly suppressing tumor recall.

Epoch-end GPU temperatures were usually `86–87°C`. Cooldown checks reduced the next epoch start to approximately `73–77°C`. The 90°C emergency limit was not crossed.

## 8. Current scientific interpretation

1. The corrected dataset and loader are usable.
2. The main problem is cross-patient generalization, not inability to optimize.
3. Global Dice hides severe patient failures.
4. Volume 104 is highly preprocessing/domain sensitive.
5. Volume 116 is persistently missed across most experiments.
6. Increasing recall through sampling alone did not improve the patient mean.
7. Recall-aware loss configurations can collapse.
8. Context, anatomical gating, and post-processing reduce false positives but can destroy recall.
9. Ground-truth-organ-assisted normalization demonstrates that useful tumor signal exists, but that pipeline is not deployable.
10. The multi-task model likely produces weak tumor probabilities for volumes 104/116 rather than merely having an incorrect binary threshold.

## 9. What should happen next

Do not open the test split.

Do not continue the existing multi-task model to epoch 15 without a new hypothesis. Its validation result plateaued after epoch 8.

The next experiment should be a validation-only probability calibration and localization diagnostic using the frozen epoch-8 multi-task checkpoint.

### Required calibration sweep

- Tumor threshold: approximately `0.10–0.60`.
- Liver threshold: approximately `0.30–0.70`.
- Liver dilation kernel: `1`, `5`, `11`, `21`, and `31`.
- Raw tumor probability.
- Predicted-liver-supported tumor probability.

### Required measurements

- Global and patient Dice.
- Volumes 104 and 116.
- Q1–Q4 detection.
- Positive predicted-empty.
- Empty-slice false positives.
- Pixel precision and recall.
- Percentage of tumor probability removed by liver support.
- Probability distributions for:
  - True tumor pixels.
  - Non-tumor liver pixels.
  - Extra-liver pixels.
  - Volumes 104 and 116.
  - Strong patients such as 108–110.

### Decision after calibration

- If lower thresholds recover volume 104/116 and Q1 detection while empty-slice FP remains ≤15%, freeze the calibrated configuration.
- If probabilities are correctly localized but too low, revise calibration or the false-negative penalty.
- If probabilities are spatially incorrect, thresholding cannot solve the problem.

## 10. Recommended training design if calibration fails

Use a true two-stage predicted-liver ROI pipeline:

1. Predict liver.
2. Create a padded bounding box from the predicted liver only.
3. Crop the CT image using that predicted ROI.
4. Resize the ROI to increase effective tumor resolution.
5. Train a dedicated tumor model inside the ROI.
6. Map tumor probabilities back into full-image coordinates.

Recommended improvements:

- Patient-balanced batches.
- Explicit tumor-negative liver slices as hard negatives.
- Controlled false-negative-sensitive Tversky/Dice mixture.
- Stability checks before a full run.
- Separate tracking of volumes 104 and 116.
- Checkpoint selection constrained by recall and false-positive guardrails.

## 11. Information requested from the next advisor

Please review this evidence and recommend:

1. The most scientifically defensible probability-calibration design.
2. Whether the epoch-8 multi-task probabilities should be calibrated globally, per volume, or by lesion-size group without causing validation overfitting.
3. How to diagnose why volumes 104 and 116 receive almost no correct prediction.
4. Whether a two-stage liver-ROI tumor model is justified.
5. The most stable loss and sampling strategy for improving intra-liver tumor recall while keeping empty-slice false positives below 15%.
6. How to structure checkpoint selection so one strong patient or large lesion cannot dominate.
7. What minimum validation evidence should be required before opening the locked test split once.

## 12. Important files for inspection

Documentation:

- `understanding the project\README.md`
- `understanding the project\01_dataset_and_provenance.md`
- `understanding the project\03_experiment_timeline.md`
- `understanding the project\05_complete_results.md`
- `understanding the project\08_current_plan.md`

Current checkpoint and results:

- `Practice\multitask_liver_tumor_outputs\multitask_best.pth`
- `Practice\multitask_liver_tumor_outputs\multitask_history.csv`
- `Practice\multitask_liver_tumor_outputs\best_validation_patient_metrics.csv`
- `Practice\multitask_liver_tumor_outputs\best_validation_per_slice.csv`
- `Practice\multitask_liver_tumor_outputs\best_validation_size_quartiles.csv`
- `Practice\multitask_liver_tumor_outputs\multitask_localization_gate_result.json`
- `Practice\multitask_liver_tumor_outputs\multitask_results_dashboard.png`
- `Practice\multitask_liver_tumor_outputs\expected_vs_generated_predictions.png`

Diagnostic comparison:

- `Practice\intensity_robustness_outputs\intensity_robustness_gate_result.json`
- `Practice\context_2_5d_outputs\context_2_5d_gate_result.json`
- `Practice\validation_3d_postprocessing_outputs\validation_3d_postprocessing_gate_result.json`

## 13. Non-negotiable safeguards

- Do not use the older `next_phase_alignment_and_overfit.ipynb`; it references invalid pairing.
- Do not compare models without stating changes in preprocessing, sampling, loss, or channels.
- Do not use ground-truth liver masks for reported validation/test preprocessing or prediction gating.
- Do not tune using the test split.
- Keep manifest and checkpoint hashes in every result.
- Keep patient-level and lesion-size metrics in every decision.


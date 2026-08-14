# Experiment Timeline

## Phase 1 — Initial EDA and dataset pipeline

Notebooks:

- `lits_eda.ipynb`
- `dataset_pipeline.ipynb`
- `unified_dataset_preparation_and_eda*.ipynb`

Activities:

- Inspected CT images, masks, class imbalance, patient volumes, and slice distributions.
- Traced raw NIfTI sources to derived PNGs.
- Found differences between image and mask geometry in the earlier pipeline.
- Established that naive directory pairing and shape checks were not sufficient.

## Phase 2 — Pairing and orientation forensics

Notebooks:

- `run_dataset_pairing_forensics.ipynb`
- `dataset_pairing_forensics.ipynb`
- `dataset_validation_and_promotion.ipynb`

Results:

- 58,638 rows and 131 volumes audited.
- All sampled source-to-derived slice regenerations matched.
- Slice offset was zero.
- 47 volumes required orientation correction.
- A corrected, manifest-driven v2 dataset was produced and spatially approved.

## Phase 3 — Verified loader and deterministic overfit

Notebook: `verified_manifest_loader_and_overfit_gate.ipynb`

Techniques:

- Strict manifest loader.
- Schema, path, hash, orientation, split, pixel-count, and binary-mask checks.
- Deterministic selection of 16 training slices.
- MobileNetV2-U-Net overfit test.
- Safe resume and thermal monitoring.

Result:

- Loader gate passed.
- Overfit gate passed.
- Best hard micro-Dice `0.816306` after 76 epochs.
- Test remained locked.

## Phase 4 — Five-epoch baseline smoke test

Notebook: `manifest_baseline_5epoch_smoke_test.ipynb`

Configuration:

- MobileNetV2-U-Net, one input channel, one tumor output.
- Focal-Dice loss.
- Batch size 8; validation batch 16.
- Learning rate `3e-4`.
- Weight decay `1e-4`.
- Positive-slice sampling weight `4`.
- Threshold `0.50`.
- No pretrained encoder.

Result:

- Gate passed.
- Epoch 5 was best.
- Validation micro-Dice `0.461613`.
- Positive-slice recall `56.05%`.
- Positive predicted-empty `27.74%`.
- Empty-slice false positives `14.42%`.

## Phase 5 — Validation geometry and volume forensics

Notebook: `validation_volume_mask_orientation_forensics.ipynb`

Activities:

- Audited 1,042 validation tumor-positive slices and 1,200 training reference slices.
- Confirmed minimum volume and slice tumor containment of `1.0`.
- Found zero slices below 99% containment.
- The automated source-metadata gate still failed and required manual review.

This phase prevented an apparent modeling problem from being accepted without checking data geometry.

## Phase 6 — Patient-aware baseline to epoch 10

Notebooks:

- `baseline_25epoch_patient_aware_training.ipynb`
- `continue_patient_aware_baseline_to_epoch10.ipynb`
- `auto_cool_continue_patient_aware_to_epoch10.ipynb`

Result:

- Best epoch `9`.
- Global micro-Dice `0.558357`.
- Mean patient Dice `0.383069`.
- Median patient Dice `0.342593`.
- Worst patient Dice approximately zero.
- Volume 104 Dice `0.197613`.
- Volume 116 Dice approximately zero.
- Decision: stop and diagnose patient/lesion failures.

## Phase 7 — Lesion-balanced sampler ablation

Notebook: `patient_lesion_balanced_sampler_ablation.ipynb`

Technique:

- Inverse-volume × lesion-quartile sampling.

Result:

- Best epoch `7`.
- Global Dice `0.524200`.
- Mean patient Dice `0.356189`.
- Volume 104 `0.152444`.
- Volume 116 `0.001856`.
- Q1 detection improved to `53.99%`.
- Positive predicted-empty `19.29%`.
- Empty-slice false positives `17.83%`.
- Decision: sampling alone was insufficient.

## Phase 8 — Recall-aware loss

Notebook: `recall_aware_focal_tversky_loss_ablation.ipynb`

Technique:

- Focal Tversky, alpha `0.40`, beta `0.60`, gamma `0.75`.
- Baseline positive-slice weight `3`.

Result:

- Collapsed to empty predictions.
- Best epoch `1`.
- Mean patient Dice approximately zero.
- Q1 detection `0%`.
- Positive predicted-empty `100%`.
- Empty-slice false positives `0%`.

## Phase 9 — Stabilized composite loss

Notebook: `stabilized_composite_loss_ablation.ipynb`

Technique:

- `0.75` Focal-Dice + `0.25` Focal-Tversky.

Result:

- Non-collapsed.
- Best epoch `5`.
- Global Dice `0.568154`.
- Mean patient Dice `0.322939`.
- Volume 104 `0.005005`.
- Volume 116 `0.051481`.
- Q1 detection `29.66%`.
- Positive predicted-empty `41.46%`.
- Empty-slice false positives `3.03%`.
- Recovered volume 116 slightly but damaged the broader patient result.

## Phase 10 — Appearance/domain forensics

Notebook: `appearance_domain_robustness_forensics.ipynb`

Results:

- Training volumes: 104.
- Validation volumes: 13.
- Sampled slices: 3,106.
- Training P95 shift threshold: `6.4022`.
- Volume 104 shift score: `6.6768`, flagged as an appearance outlier.
- Volume 116 shift score: `3.7398`, not above the same threshold.

## Phase 11 — Organ-assisted intensity robustness

Notebook: `organ_normalized_intensity_robustness_ablation.ipynb`

Technique:

- Robust median/IQR statistics within the ground-truth organ mask.
- Clip robust z-score to ±3.
- Gamma augmentation `0.85–1.15`.
- Bounded Gaussian noise up to standard deviation `0.025`.
- Focal-Dice loss and positive-slice weight `3`.

Result:

- Best epoch `8`.
- Global Dice `0.546132`.
- Mean patient Dice `0.406915`.
- Volume 104 `0.586500`.
- Volume 116 `0.026782`.
- Q1 detection `60.08%`.
- Positive predicted-empty `8.25%`.
- Empty-slice false positives `47.42%`.

Interpretation:

- Strong diagnostic improvement in recall and volume 104.
- Failed false-positive and volume 116 guardrails.
- Not a deployable preprocessing pipeline because ground-truth organ masks contributed to validation normalization.

## Phase 12 — Adjacent-slice 2.5D context

Notebook: `adjacent_slice_2_5d_context_ablation.ipynb`

Technique:

- Three input channels: previous, current, and next axial slice.

Result:

- Best epoch `10`.
- Global Dice `0.553527`.
- Mean patient Dice `0.355938`.
- Volume 104 `0.000636`.
- Volume 116 `0.000932`.
- Q1 detection `25.48%`.
- Positive predicted-empty `42.99%`.
- Empty-slice false positives `2.37%`.

Context reduced false positives but destroyed recall for key patients.

## Phase 13 — Validation-only 3D post-processing

Notebook: `validation_3d_postprocessing_ablation.ipynb`

Techniques:

- Raw thresholds.
- 3D connected components.
- Hysteresis thresholds.
- Minimum component volume.
- Minimum axial span.

Result:

- Best configuration was unchanged `raw_t050`.
- Mean patient Dice `0.406964`.
- Volume 104 `0.586555`.
- Volume 116 `0.026794`.
- Q1 detection `60.08%`.
- Positive predicted-empty `8.25%`.
- Empty-slice false positives `47.43%`.
- Every cleanup rule lost patient Dice before reaching the false-positive target.

## Phase 14 — Multi-task liver and tumor localization

Notebooks:

- `multitask_liver_tumor_localization.ipynb`
- `multitask_liver_tumor_epoch10_continuation.ipynb`

Techniques:

- Shared MobileNetV2-U-Net with two outputs: liver and tumor.
- Image-only robust normalization.
- Ground-truth liver used as training supervision only.
- Tumor gated by dilated predicted-liver support.
- Warm start from the intensity checkpoint.
- Loss weights: tumor `0.72`, liver `0.23`, containment `0.05`.
- Learning rate `1e-4`.
- Positive-slice weight `3`.

Result:

- Completed epoch-10 decision point.
- Best epoch `8`.
- Mean patient Dice `0.332854`.
- Liver Dice `0.886322`.
- Volume 104 and 116 Dice approximately zero.
- Q1 detection `28.14%`.
- Positive predicted-empty `46.16%`.
- Empty-slice false positives `3.25%`.
- Raw and liver-gated tumor predictions were identical; the gate removed zero predicted pixels.

Decision: fail, keep test locked, and calibrate probabilities before another training run.


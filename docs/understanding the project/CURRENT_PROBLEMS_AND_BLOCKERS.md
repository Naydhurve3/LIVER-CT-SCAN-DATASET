# Current Problems and Blockers — Liver Tumor Segmentation

This document is a problem-focused briefing for another chat, researcher, or collaborator. It explains what is currently failing, what evidence supports each problem, what has already been tried, and what should be investigated next.

## 1. Short description of the situation

The corrected dataset, manifest loader, masks, and training pipeline have passed integrity and small-subset overfit tests. The main unresolved problem is **generalization across validation patients**.

Models can obtain reasonable aggregate Dice while completely missing tumors in particular patients. Attempts to improve recall often increase false positives, while attempts to control false positives suppress true tumors. Volumes 104 and 116 are the most persistent failures.

The current multi-task liver/tumor model completed its epoch-10 validation decision point and failed six of seven targets. The test split remains locked.

## 2. Authoritative dataset

- Dataset: LiTS liver CT.
- Corrected build: `build_corrected_20260713_214847_v2`.
- Build path:
  `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- Manifest:
  `manifests\slice_manifest.csv`
- Manifest SHA-256:
  `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- Total slices: `58,638`.
- Total volumes/patients: `131`.
- Image and mask size: `256 × 256`.
- Target: binary tumor.
- Liver/organ mask: stored separately.
- All rows are verified.
- All spatial statuses are approved.

### Patient split

| Split | Slices | Patients | Tumor-positive slices |
|---|---:|---:|---:|
| Train | 40,667 | 104 | 4,930 |
| Validation | 10,685 | 13 | 1,042 |
| Test | 7,286 | 14 | 1,197 |

The splits are patient-disjoint. Test images have not been accessed in any saved gate result.

## 3. Data problems already resolved

These are not believed to be the current modeling blocker, but another chat must know they occurred.

### Incorrect image-mask orientation

The earlier dataset build contained 47 volumes with incompatible in-plane CT/mask orientation.

- Volumes `83–99` and `101–130` required 180-degree correction.
- Volumes `0–82` and `100` used identity orientation.
- Slice offsets were correct.
- File names and smooth axial continuity did not reveal the in-plane error.

This was corrected in the v2 build.

### Incorrect tumor-burden denominator

An earlier audit divided `256×256` mask pixels by `512×512`, under-reporting tumor burden fourfold. The calculation now uses the actual mask dimensions.

### Ambiguous mask semantics

Older documentation could be read as liver-plus-tumor foreground. The current contract is:

- Tumor mask is the segmentation target.
- Organ/liver mask is separate.
- Do not mix these semantics when comparing experiments.

## 4. Current primary problem: patient-level generalization

The strongest deployable-style patient-aware baseline obtained:

- Global micro-Dice: `0.558357`.
- Mean patient Dice: `0.383069`.
- Median patient Dice: `0.342593`.
- Worst patient Dice: approximately zero.
- Volume 104 Dice: `0.197613`.
- Volume 116 Dice: approximately zero.

This means the aggregate score hides complete failure on at least one important patient.

### Why this matters

A segmentation model cannot be accepted because it performs well on large or easy lesions while producing empty masks for entire patients.

### Required monitoring

- Mean, median, and worst positive-patient Dice.
- Per-patient Dice.
- Volumes 104 and 116.
- Small-lesion detection.
- Positive-slice empty predictions.
- Empty-slice false positives.

## 5. Current primary problem: volume 104

Volume 104 is highly sensitive to preprocessing.

Observed Dice:

| Experiment | Volume 104 Dice |
|---|---:|
| Patient-aware baseline | `0.197613` |
| Lesion-balanced sampler | `0.152444` |
| Composite loss | `0.005005` |
| Organ-assisted normalization | `0.586500` |
| 2.5D context | `0.000636` |
| Multi-task image-only model | approximately zero |

Appearance forensics found:

- Training P95 shift threshold: `6.4022`.
- Volume 104 shift score: `6.6768`.
- It was flagged as an appearance-domain outlier.

### Current interpretation

The tumor is segmentable when ground-truth-organ-based normalization is used, but performance collapses under image-only normalization and other representations.

### Unresolved questions

- Are tumor probabilities spatially correct but below threshold?
- Does global/non-zero-pixel normalization distort this patient's liver contrast?
- Are scanner, reconstruction, intensity, or field-of-view characteristics different?
- Would a predicted-liver ROI permit robust local normalization without using ground-truth liver?

## 6. Current primary problem: volume 116

Volume 116 is persistently missed despite having a large tumor burden.

Observed Dice:

| Experiment | Volume 116 Dice |
|---|---:|
| Patient-aware baseline | approximately zero |
| Lesion-balanced sampler | `0.001856` |
| Composite loss | `0.051481` |
| Organ-assisted normalization | `0.026782` |
| 2.5D context | `0.000932` |
| Multi-task image-only model | approximately zero |

At the best multi-task epoch:

- True tumor pixels: `152,763`.
- Predicted pixels: `534`.
- Positive-slice recall: `0%`.
- Positive predicted-empty: `96.24%`.

Appearance shift:

- Volume 116 shift score: `3.7398`.
- This was below the training P95 threshold.

### Current interpretation

Volume 116 is not simply a tiny-lesion problem and is not explained by the same aggregate appearance-shift signal as volume 104.

### Unresolved questions

- Is its tumor phenotype unusually diffuse or low contrast?
- Is the tumor label spatially valid but visually unlike training examples?
- Does preprocessing remove relevant contrast?
- Are model probabilities misplaced or uniformly low?
- Is volume 116 represented poorly by slice-weighted training?

## 7. Current precision-versus-recall conflict

The experiments show a repeated tradeoff:

### High recall but excessive false positives

Organ-assisted intensity model:

- Mean patient Dice: `0.406915`.
- Q1 detection: `60.08%`.
- Positive predicted-empty: `8.25%`.
- Empty-slice false positives: `47.42%`.

### Low false positives but excessive missed tumors

Multi-task model:

- Mean patient Dice: `0.332854`.
- Q1 detection: `28.14%`.
- Positive predicted-empty: `46.16%`.
- Empty-slice false positives: `3.25%`.

2.5D model:

- Q1 detection: `25.48%`.
- Positive predicted-empty: `42.99%`.
- Empty-slice false positives: `2.37%`.

### Current interpretation

The model family can move between recall-heavy and suppression-heavy solutions, but no tested configuration achieves both adequate recall and controlled false positives.

## 8. Current multi-task model problem

The multi-task model predicts liver and tumor using a shared MobileNetV2-U-Net.

Parameters:

- One image input channel.
- Two output channels: liver and tumor.
- Image-only robust normalization.
- Tumor loss weight: `0.72`.
- Liver loss weight: `0.23`.
- Containment loss weight: `0.05`.
- Learning rate: `1e-4`.
- Tumor threshold: `0.50`.
- Liver threshold: `0.50`.
- Liver dilation kernel: `31`.
- Positive-slice sampling weight: `3`.

Best epoch: `8`.

Results:

- Mean patient Dice: `0.332854`.
- Mean liver Dice: `0.886322`.
- Volume 104: approximately zero.
- Volume 116: approximately zero.
- Q1 detection: `28.14%`.
- Positive predicted-empty: `46.16%`.
- Empty-slice false positives: `3.25%`.
- Passed targets: `1/7`.

### Critical finding: liver gating did nothing

Raw tumor predictions and predicted-liver-gated tumor predictions were exactly identical:

- Maximum Dice difference: `0`.
- Maximum predicted-pixel difference: `0`.
- Maximum empty-slice-FP difference: `0`.

Every predicted tumor pixel was already inside the dilated predicted-liver support.

### Meaning

- Extra-liver false positives are not the main remaining problem.
- The failures are occurring inside anatomically plausible liver regions.
- Merely changing the binary liver gate is unlikely to solve the tumor discrimination problem.
- The dilation kernel of 31 may be too permissive, but even a smaller gate cannot create missing tumor probability.

## 9. Small-lesion and large-lesion problems

Best multi-task size results:

| Size group | Detection | Mean Dice | Predicted empty |
|---|---:|---:|---:|
| Q1 smallest | `28.14%` | `0.1556` | `66.16%` |
| Q2 | `58.14%` | `0.3273` | `36.43%` |
| Q3 | `61.15%` | `0.4274` | `37.31%` |
| Q4 largest | `49.81%` | `0.3250` | `44.44%` |

The Q4 median Dice was approximately zero. Therefore the problem is not only small lesions. Large tumor volumes 104 and 116 heavily influence the Q4 failure.

## 10. Ground-truth-organ preprocessing problem

The strongest result used median/IQR statistics calculated inside the ground-truth liver mask.

That experiment proved that:

- useful tumor signal exists;
- volume 104 can be segmented;
- small-lesion recall can improve.

But the pipeline is not deployable because ground-truth liver masks are not available during real inference.

Another chat must not recommend using validation/test ground-truth liver masks for:

- normalization;
- cropping;
- gating;
- threshold selection.

Predicted liver may be used, provided the complete inference pipeline is evaluated.

## 11. Sampling problem

Tumor-positive slices are rare relative to all slices.

Positive weighting helps recall, but can encourage false tumor predictions on tumor-negative liver slices.

The inverse-volume × lesion-quartile sampler:

- improved Q1 detection to `53.99%`;
- reduced mean patient Dice to `0.356189`;
- did not recover volumes 104 or 116.

The current multi-task positive-slice weight is `3`.

### Unresolved question

What batch composition balances:

- tumor-positive slices;
- tumor-negative liver slices;
- empty-background slices;
- difficult patients;
- small lesions;
- large diffuse lesions?

## 12. Loss-function problem

### Pure Focal-Tversky

- Alpha `0.40`.
- Beta `0.60`.
- Gamma `0.75`.
- Collapsed to empty predictions.
- Positive predicted-empty: `100%`.

### Composite loss

- 75% Focal-Dice.
- 25% Focal-Tversky.
- Did not collapse.
- Recovered volume 116 to `0.051481`.
- Mean patient Dice fell to `0.322939`.

### Current problem

A stronger false-negative penalty might improve recall, but previous recall-aware losses were unstable or redistributed performance.

Any future loss must first pass:

- finite-loss checks;
- gradient checks;
- 16-slice overfit;
- a short validation smoke test;
- empty-prediction monitoring.

## 13. Post-processing problem

Validation-only post-processing tested:

- thresholds `0.20`, `0.30`, and `0.50`;
- 3D connected components;
- hysteresis thresholds;
- minimum component sizes;
- minimum axial spans.

The best option was unchanged raw threshold `0.50`.

Every cleanup configuration lost patient Dice before meeting the false-positive requirement.

Conclusion:

Post-processing cannot create missing tumor probability and should not be the next major intervention.

## 14. Threshold-calibration gap

The current multi-task model is evaluated primarily at threshold `0.50`.

Because the model is over-suppressed, the highest-value unanswered question is whether true tumor probabilities exist below `0.50`.

Required validation-only study:

- Tumor thresholds approximately `0.10–0.60`.
- Liver thresholds approximately `0.30–0.70`.
- Liver dilation kernels `1`, `5`, `11`, `21`, `31`.
- Raw and liver-supported tumor probabilities.

Required analyses:

- Precision-recall curves.
- Mean patient Dice versus empty-slice FP.
- Q1 detection versus positive-empty rate.
- Volume 104 and 116 Dice by threshold.
- True-tumor probability histograms.
- Liver-background probability histograms.
- Extra-liver probability histograms.
- Percentage of tumor probability removed by liver support.

This should be done before another training run.

## 15. Runtime and engineering problems already encountered

### Matplotlib API incompatibility

Error:

`Axes.boxplot() got an unexpected keyword argument 'labels'`

Resolved by using a compatible tick-label approach.

### RNG restoration

Error:

`RNG state must be a torch.ByteTensor`

Resolved by converting restored RNG tensors to detached CPU `uint8`.

### Missing thermal constant

Error:

`MAX_GPU_TEMP_C is not defined`

Resolved by using consistent named thermal constants.

### Non-finite loss

Error:

`FloatingPointError: Non-finite loss`

Occurred around epochs 7/8 in earlier experiments.

Controls added:

- finite-loss checks;
- mixed-precision handling;
- gradient clipping at norm 5;
- stable checkpoints;
- exact resume.

### 2.5D channel mismatch

Error:

The 3-channel model received a 1-channel review batch.

Resolved by using a subset of the existing adjacent-slice validation dataset and displaying its center channel.

### GPU heat

Observed:

- Epoch-end temperature commonly `86–87°C`.
- Desired start temperature below `84°C`.
- Emergency stop at `90°C`.
- Cooldown usually reduced start temperature to `73–77°C`.

Controls:

- 30-second cooling polls.
- Maximum 30-minute bounded wait.
- Per-epoch checkpoints.
- Thermal CSV.

Future options:

- Better airflow/cooling.
- Batch size 4.
- Gradient accumulation to retain effective batch size.

## 16. What has been ruled out

The evidence currently argues against:

- Returning to the old invalid-pairing notebook.
- Assuming a file-pairing or slice-offset error in the corrected v2 build.
- Accepting aggregate Dice without patient metrics.
- Using pure Focal-Tversky again unchanged.
- Solving the problem only through lesion-balanced sampling.
- Solving missing tumor signal through connected-component filtering.
- Assuming 2.5D context automatically improves tumor detection.
- Using ground-truth liver masks during reported validation/test inference.
- Continuing the multi-task model beyond epoch 10 without a new hypothesis.

## 17. Recommended next decision sequence

### Step 1 — Frozen checkpoint probability calibration

Use:

`Practice\multitask_liver_tumor_outputs\multitask_best.pth`

Do not train.

Determine whether volumes 104/116 and Q1 lesions have:

- correctly localized low probabilities;
- incorrectly localized probabilities;
- no tumor signal.

### Step 2 — Decide from calibration

If lower threshold recovers recall with empty-slice FP ≤15%:

- freeze the calibrated threshold/gating configuration;
- rerun the complete validation gate;
- do not tune per patient.

If probabilities are correctly located but too low:

- investigate calibration and false-negative-sensitive loss;
- use one global validation-derived threshold.

If probabilities are spatially wrong:

- thresholding will not help;
- proceed to a predicted-liver ROI tumor model.

### Step 3 — Two-stage model if needed

1. Predict liver.
2. Create a padded predicted-liver bounding box.
3. Crop and resize the CT ROI.
4. Segment tumor at higher effective resolution.
5. Map probability back to the original image.
6. Evaluate the complete predicted-liver pipeline.

Potential training improvements:

- patient-balanced batches;
- hard-negative tumor-free liver slices;
- small-lesion oversampling with caps;
- false-negative-sensitive stable loss;
- constrained checkpoint selection.

## 18. Checkpoint-selection problem

Current experiments frequently save the highest mean patient Dice, even if important guardrails fail.

Future checkpoint eligibility should require:

- Empty-slice FP ≤ `15%`.
- Positive predicted-empty ≤ `20%`.
- Q1 detection ≥ `45%`.
- Volume 104 Dice ≥ `0.50`.
- Volume 116 Dice ≥ `0.05`.
- Liver Dice ≥ `0.90` for multi-task models.

Eligible checkpoints can then be ranked by mean patient Dice.

If no checkpoint is eligible, report:

- targets passed;
- Pareto frontier;
- failure reason;
- no-go decision.

## 19. Test split status

The test split is still locked.

Do not open it until:

1. One complete validation configuration passes every declared target.
2. Checkpoint, preprocessing, thresholds, ROI rules, and metric code are frozen.
3. Manifest and checkpoint hashes are recorded.
4. The test is evaluated once without tuning.

## 20. Questions another chat should answer

Please advise on:

1. The exact probability-calibration design that minimizes validation overfitting.
2. Whether global temperature scaling, thresholding, or another calibration method is most appropriate for segmentation probabilities.
3. How to diagnose volumes 104 and 116 without patient-specific tuning.
4. Whether the predicted-liver ROI approach is justified by the current evidence.
5. The best stable loss for increasing intra-liver tumor recall without returning to 47% empty-slice false positives.
6. An appropriate patient-balanced/hard-negative batch design.
7. Whether normalization should use global image statistics, predicted-liver statistics, or a learned normalization mechanism.
8. How to select checkpoints with multiple guardrails.
9. What validation evidence is sufficient before the one-time test evaluation.

## 21. Files to provide to another chat

Primary problem summary:

- `understanding the project\CURRENT_PROBLEMS_AND_BLOCKERS.md`
- `understanding the project\CHAT_HANDOFF_SUMMARY.md`

Detailed documentation:

- `understanding the project\01_dataset_and_provenance.md`
- `understanding the project\03_experiment_timeline.md`
- `understanding the project\05_complete_results.md`
- `understanding the project\08_current_plan.md`

Current outputs:

- `Practice\multitask_liver_tumor_outputs\multitask_localization_gate_result.json`
- `Practice\multitask_liver_tumor_outputs\multitask_history.csv`
- `Practice\multitask_liver_tumor_outputs\best_validation_patient_metrics.csv`
- `Practice\multitask_liver_tumor_outputs\best_validation_per_slice.csv`
- `Practice\multitask_liver_tumor_outputs\best_validation_size_quartiles.csv`
- `Practice\multitask_liver_tumor_outputs\multitask_results_dashboard.png`
- `Practice\multitask_liver_tumor_outputs\expected_vs_generated_predictions.png`

Comparison outputs:

- `Practice\intensity_robustness_outputs\intensity_robustness_gate_result.json`
- `Practice\context_2_5d_outputs\context_2_5d_gate_result.json`
- `Practice\validation_3d_postprocessing_outputs\validation_3d_postprocessing_gate_result.json`


# Current Interpretation and Next Plan

## Current decision

The epoch-10 multi-task gate failed. Do not:

- Continue automatically to epoch 15.
- Open the test split.
- Add another long training run without first examining probabilities.
- Claim the organ-assisted intensity result as deployable.

## What is known

1. The corrected loader and masks can be fitted: the 16-slice gate reached Dice `0.8163`.
2. A stable baseline generalizes unevenly: mean patient Dice `0.3831`, but volume 116 is missed.
3. Appearance normalization can recover volume 104 and small-lesion recall.
4. That same run produces excessive empty-slice false positives and depends on ground-truth organ statistics.
5. 2.5D context and multi-task learning strongly suppress false positives but also suppress true tumors.
6. Volume 104 is sensitive to preprocessing/domain changes.
7. Volume 116 is persistently difficult and is not explained by the same appearance-shift score as volume 104.
8. Multi-task liver gating is currently a no-op because raw and gated predictions are identical.

## Immediate next experiment: probability calibration

Use the frozen epoch-8 multi-task checkpoint and validation data only.

Sweep:

- Tumor threshold: approximately `0.10–0.60`.
- Liver threshold: approximately `0.30–0.70`.
- Liver dilation kernels: `0/1`, `5`, `11`, `21`, and current `31`.
- Raw tumor probability and predicted-liver-supported probability.

Record for every combination:

- Global Dice.
- Mean/median/worst patient Dice.
- Volume 104 and 116 Dice.
- Q1–Q4 detection and Dice.
- Positive predicted-empty percentage.
- Empty-slice false-positive percentage.
- Pixel precision/recall.
- Percentage of tumor probability removed by liver support.
- Per-volume probability distributions on true tumor, liver background, and extra-liver background.

Required graphs:

- Precision-recall curve.
- Mean patient Dice versus empty-slice FP frontier.
- Q1 detection versus positive-empty frontier.
- Volume 104/116 Dice by threshold.
- Probability histograms for volumes 104, 116, and strong patients.
- Raw-versus-gated removed-pixel analysis.
- Expected-versus-actual target matrix.

Decision:

- If a calibrated threshold recovers recall while keeping empty-slice FP ≤15%, freeze it.
- If volumes 104/116 probabilities are correctly located but too low, revise calibration/loss.
- If probabilities are spatially misplaced, calibration cannot solve the issue; proceed to ROI modeling.

## Next training design if calibration fails

### Two-stage predicted-liver ROI model

1. Train or freeze a liver-localization model.
2. Generate a padded predicted-liver bounding box.
3. Crop the CT image using only predicted liver at validation/test time.
4. Resize the ROI to a higher effective resolution.
5. Train a tumor model inside the predicted ROI.
6. Map the tumor probability back to full-image coordinates.

Recommended safeguards:

- Patient-balanced batches.
- Explicit tumor-negative liver slices as hard negatives.
- False-negative-sensitive but stability-tested Tversky/Dice mixture.
- Separate monitoring of V104, V116, Q1 detection, and empty-slice FP.
- Constrained checkpoint selection rather than mean patient Dice alone.

## Checkpoint-selection improvement

A checkpoint should first satisfy guardrails:

- Empty-slice FP ≤15%.
- Positive predicted-empty ≤20%.
- Liver Dice ≥0.90 for multi-task models.
- Q1 detection ≥45%.
- Volume 104 ≥0.50.
- Volume 116 ≥0.05.

Only eligible checkpoints should then be ranked by mean patient Dice. If no checkpoint is eligible, report the number of targets passed and the Pareto frontier rather than calling one “best” without qualification.

## Thermal plan

- Start epochs below 84°C.
- Keep the 90°C emergency stop.
- Retain 30-second bounded cooling checks.
- Consider batch size 4 with gradient accumulation if temperatures remain 86–87°C at epoch end.
- Improve airflow/cooling before another long experiment.

## Test authorization

The test split remains locked until:

1. One complete validation configuration passes all declared targets.
2. Model checkpoint, preprocessing, thresholds, ROI rules, and metrics are frozen.
3. A final manifest/checkpoint hash record is written.
4. The test is evaluated once without tuning.


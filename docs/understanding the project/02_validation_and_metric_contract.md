# Validation and Metric Contract

## Why aggregate Dice is insufficient

The validation distribution contains large differences in tumor burden. A high global/micro Dice can be driven by large lesions and a few strong patients while small lesions or entire patients are missed.

Every serious decision therefore uses:

- Global pixel micro-Dice.
- Mean, median, and worst positive-patient micro-Dice.
- Per-patient Dice, especially volumes 104 and 116.
- Pixel precision and recall.
- Positive-slice recall.
- Positive slices predicted completely empty.
- Empty-slice false-positive rate.
- Lesion-size-stratified detection and Dice.
- Threshold sensitivity.
- Expected-versus-actual targets.

## Current lesion-size evaluation

Positive validation slices are divided into four equal-frequency quartiles using validation tumor-pixel count for reporting. Earlier baseline planning also recorded training-derived reference groups:

- Tiny: `1–51` pixels.
- Small: `52–196` pixels.
- Medium: `197–699` pixels.
- Large: `≥700` pixels.

Do not silently interchange training-derived fixed groups and validation `qcut` quartiles. Label which definition is used.

## Core formulas

Hard Dice:

`Dice = (2 × intersection + epsilon) / (predicted pixels + true pixels + epsilon)`

Patient micro-Dice is computed after summing intersection, predicted pixels, and true pixels over all slices in a volume.

Positive-slice recall:

`positive slices with any true-positive overlap / all tumor-positive slices`

Positive predicted-empty percentage:

`tumor-positive slices with zero predicted pixels / all tumor-positive slices × 100`

Empty-slice false-positive percentage:

`tumor-negative slices with any predicted pixel / all tumor-negative slices × 100`

## Standard thresholds and sweeps

- Default hard tumor threshold: `0.50`.
- Standard diagnostic sweep used in several notebooks: `0.30–0.90` in `0.05` steps.
- The next calibration must extend lower, approximately `0.10–0.60`, because the multi-task model appears over-suppressed at `0.50`.

For the baseline/intensity experiments, threshold changes alone were weak: in the 2.5D run, global Dice varied only from about `0.557` at threshold `0.30` to `0.543` at threshold `0.90`, while recall stayed low.

## Multi-task epoch-10 targets

| Metric | Required direction | Target |
|---|---|---:|
| Mean positive-patient Dice | Higher | `0.406915` |
| Volume 104 Dice | Higher | `0.50` |
| Volume 116 Dice | Higher | `0.05` |
| Q1 smallest-lesion detection | Higher | `45%` |
| Positive predicted-empty rate | Lower | `20%` |
| Empty-slice false-positive rate | Lower | `15%` |
| Mean liver Dice | Higher | `0.90` |

## Overfit gate

- Selected training slices: `16`.
- Maximum epochs: `150`.
- Actual completed epochs: `76`.
- Required hard micro-Dice: `0.80`.
- Achieved hard micro-Dice: `0.816306`.
- Threshold: `0.50`.
- Result: pass.

Passing the overfit gate proved that the corrected loader, masks, model, optimizer, and loss could fit a small deterministic subset. It did not prove validation generalization.

## Test-lock contract

- Test patients are disjoint from train and validation patients.
- Test images were not accessed in any saved gate result.
- A test run is allowed only after all declared validation targets pass.
- Once authorized, checkpoint, preprocessing, threshold, post-processing, and metric code must be frozen before opening test data.
- Test results must not be used to choose a threshold or modify the model.

## Comparison rules

Do not make causal claims between experiments if any of these changed:

- Loader or file pairing.
- Dataset build or manifest hash.
- Mask semantics.
- Image normalization.
- Image channels or adjacent-slice context.
- Sampling distribution.
- Loss.
- Patient aggregation.
- Threshold or post-processing.

Such runs may still be compared descriptively, but the changed factor and limitation must be stated.


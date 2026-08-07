# Limitations and Failure Analysis

## Formal outcome

Formal model acceptance failed solely because minimum positive-patient Dice was below 0.01. Aggregate and integrity targets passed.

## V121

- Patient Dice: 0.0000000000
- Truth pixels: 526
- Frozen ROI containment: 100.0%
- Maximum cached fused score in truth: 0.00000000
- Truth pixels at or above 0.70: 0
- Interpretation: recognition failure, not ROI clipping.

## Other weak cases

V120 achieved high recall but excessive false-positive volume, producing Dice 0.1123. V127 contained only 73 tumour pixels and achieved partial overlap with Dice 0.0826. Four of 13 positive patients were below 0.3329.

## Lesion-size limitation

The smallest train-derived volume quartile had 60.3% detection and mean matched Dice 0.238, markedly below larger strata.

## Prohibited response

Do not adjust the threshold, checkpoints, fusion, ROI, post-processing or training based on these test failures. Any follow-up model must start a new versioned study and use a new untouched evaluation cohort.

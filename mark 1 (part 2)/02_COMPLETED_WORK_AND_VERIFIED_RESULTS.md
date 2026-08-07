# Completed Work and Verified Results

## Summary

Ten major steps are complete. The latest controlling result is Mark 4E, which temporarily passes all six validation continuation targets using fixed checkpoint fusion. Older results remain useful for diagnosis but do not override Mark 4D/4E metric definitions.

## Completed steps

1. Corrected dataset provenance, manifest, split, geometry, mask and orientation audit.
2. Verified manifest loader, test lock and deterministic overfit proof.
3. Initial Mark 1 probability, contrast and localization diagnostic.
4. Mark 2 predicted-liver ROI and multi-window feasibility analysis.
5. Mark 3 ROI geometry and deterministic multi-window overfit gate.
6. Mark 4 five-epoch two-stage ROI validation smoke training.
7. Mark 4B global probability-threshold and failure diagnostic.
8. Mark 4C two-channel and recall-loss controlled ablation.
9. Mark 4D metric reconciliation and V116 localization diagnostic.
10. Mark 4E checkpoint-fusion validation gate.

## Pipeline proof gates

- Earlier corrected-pipeline 16-slice overfit: hard micro-Dice `0.816306`.
- Mark 3 ROI overfit: hard micro-Dice `0.900557` after 17 epochs.
- Mark 3 minimum tumour containment: `1.0`.
- Mark 3 minimum round-trip Dice: `1.0`.
- These prove pipeline fit and geometry only, not generalization.

## Mark 4 and Mark 4B reference

The one-channel broad-window ROI control at epoch 5 achieved approximately:

- mean positive-patient Dice `0.3639`;
- V104 Dice `0.0672`;
- V116 Dice `0.0108`;
- Q1 detection `42.59%`;
- positive predicted-empty `36.85%`;
- empty-slice false positives `3.42%`.

Mark 4B showed that thresholds `0.05-0.70` could not reduce positive predicted-empty below 35% while retaining the other control properties. Threshold-only repair was rejected.

## Mark 4C controlled ablation

- Two-channel broad+liver-window input collapsed on tumour-positive patients and was rejected.
- Recall-loss improved recall-related metrics but damaged V116.
- A metric inconsistency was found: experimental mean patient Dice initially included four tumour-empty validation patients while the reused control used nine tumour-positive patients.
- Do not use the original Mark 4C experimental mean-Dice values as authoritative.

## Mark 4D corrected checkpoint comparison

Mark 4D used the same nine tumour-positive patients for both checkpoints.

Recall-loss at threshold `0.60`:

- mean positive-patient Dice `0.376588`;
- V104 Dice `0.100240`;
- V116 Dice `0.000712`;
- Q1 detection `47.91%`;
- positive predicted-empty `30.52%`;
- empty-slice false positives `4.79%`;
- targets passed: `5/6`.

V116 diagnosis:

- all `152,763` tumour pixels are inside the frozen ROI;
- median truth-region probability is zero in every lesion-size quartile;
- recall-loss detects about `3.0%` of V116 positive slices at threshold 0.50;
- lowering threshold to 0.05 still gives only about `0.00145` V116 Dice;
- failure is localization/domain response, not simple ROI clipping or binary calibration.

## Mark 4E authoritative temporary pass

Selected inference policy:

`fused_probability = maximum(control_probability, recall_loss_probability)`

Selected threshold: `0.70`.

| Metric | Actual | Temporary target | Status |
|---|---:|---:|---|
| Mean Dice over nine tumour-positive patients | 0.377087 | >=0.3329 | Pass |
| V104 Dice | 0.116627 | >=0.05 | Pass |
| V116 Dice | 0.010474 | >=0.01 | Pass |
| Q1 detection | 50.57% | >=35% | Pass |
| Positive predicted-empty | 27.45% | <=35% | Pass |
| Empty-slice false positives | 5.55% | <=20% | Pass |

Other passing fusion policies were arithmetic mean, 75% control/25% recall, and 25% control/75% recall. Maximum fusion was selected by the predeclared highest-mean-Dice rule.

## Interpretation limits

- V116 passes by only about `0.00047` Dice above the temporary target.
- Multiple policies and thresholds were searched on validation, so a bounded confirmation/freeze stage is still needed.
- The achieved metrics do not meet the older aspirational final targets documented in `understanding the project/`.
- Mark 4E does not authorize test access.
- No final generalization claim or research-paper performance claim is permitted yet.


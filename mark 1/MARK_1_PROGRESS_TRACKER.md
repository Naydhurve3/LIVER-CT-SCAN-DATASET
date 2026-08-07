# Mark 1 Progress Tracker

Last updated: 2026-07-31 after Mark 4C controlled ablation.

## Current status

- **Overall research workflow:** approximately **70% complete**.
- **Model-development and validation phase:** approximately **65% complete**.
- **Final evaluation and reporting:** **not started**, because the test split remains locked.
- **Current decision:** Mark 4C is complete and neither ablation passed. Retain the Mark 4 control as the reference; diagnose sampling/architecture before further full training.

The percentages describe completion of the planned workflow, not expected model accuracy.

## Completed work

1. Dataset provenance, corrected manifest, split audit, geometry, labels, and preprocessing checks.
2. Deterministic loader checks and 16-slice overfit gate.
3. Initial Mark 1 modelling and validation diagnostics.
4. Failure analysis for weak patients, small lesions, empty predictions, and false positives.
5. Two-stage liver-ROI tumour segmentation smoke experiment through epoch 5.
6. Mark 4B probability cache, global threshold sweep, patient diagnostics, localization panels, and bootstrap uncertainty.
7. Test-lock verification: Mark 4B accessed no test images.
8. Mark 4C five-epoch two-channel and recall-loss ablations, including saved best checkpoints and arm comparison.

## Mark 4C verified result

Both experimental arms completed training and their checkpoints were saved. A later visualization-only cell failed because the Mark 4 control patient file used `micro_dice` rather than `dice`; the notebook has been patched and the training does not need to be repeated.

| Arm | Best epoch | Mean patient Dice | V104 Dice | V116 Dice | Q1 detection | Positive empty | Empty FP | Targets passed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 5 | 0.3639 | 0.0672 | 0.0108 | 42.59% | 36.85% | 3.42% | 5/6 |
| Two-channel | 1 | 0.0837 | approximately 0 | 0.0843 | 0.00% | 60.46% | 5.82% | 2/6 |
| Recall loss | 2 | 0.2603 | 0.0091 | 0.0019 | 39.54% | 35.70% | 3.44% | 2/6 |

No arm passed the complete continuation gate. The control remains the strongest reference. The two-channel arm degraded sharply, while recall weighting alone moved positive-empty closer to its target but damaged patient Dice and the two focus patients.

## Mark 4B verified result

The notebook executed all 11 code cells without an error. Thresholds from 0.05 to 0.70 were evaluated. No threshold passed the temporary continuation gate.

| Metric | Selected threshold 0.60 | Temporary target | Status |
|---|---:|---:|---|
| Mean patient Dice | 0.3646 | >= 0.3329 | Pass |
| Volume 104 Dice | 0.0646 | >= 0.0500 | Pass |
| Volume 116 Dice | 0.0104 | >= 0.0100 | Pass |
| Q1 lesion detection | 42.59% | >= 35.00% | Pass |
| Positive slices predicted empty | 37.04% | <= 35.00% | **Fail** |
| Empty-slice false positive rate | 3.40% | <= 20.00% | Pass |

At threshold 0.50, positive predicted-empty was 36.85%. Across the full sweep it stayed between 36.28% and 37.04%, so threshold calibration cannot solve the remaining recall/localization failure. Increasing the threshold improves precision and mean patient Dice slightly but worsens recall and the weak-patient Dice values.

Bootstrap uncertainty at threshold 0.60 is wide because validation contains nine patients: mean-patient Dice 95% bootstrap interval is approximately 0.189-0.531. Results are adequate for an experiment gate, not a final performance claim.

## Next step: Mark 4C bounded ablation

### Question

Can richer CT contrast or a recall-focused objective reduce positive predicted-empty slices without damaging empty-slice specificity?

### Experiments

Use the identical train/validation split, ROI boxes, model family, seed, evaluation code, and epoch budget. Change one factor at a time:

1. **Control:** reproduce the current single-channel broad-window configuration.
2. **Two-channel input:** broad CT window plus a liver/soft-tissue window derived from source NIfTI intensities.
3. **Recall-focused objective:** current input with a bounded increase in positive/tumour weighting or Tversky-style false-negative penalty.

Do not combine the two-channel input and objective change in the first comparison; otherwise the driver cannot be identified.

### Budget and gate

- Run a short deterministic smoke/overfit check for each changed configuration.
- Train each surviving arm for the same bounded budget, initially five epochs.
- Evaluate only the frozen validation population.
- A candidate may continue only if all temporary targets pass, especially positive predicted-empty <= 35%, while empty-slice false positives remain <= 20%.
- Select one arm before any longer continuation. Do not access the test split.

### Required outputs

- Configuration and provenance JSON, including manifest and checkpoint hashes.
- Per-epoch training/validation history.
- Patient-level metrics and temporary/final gate table.
- Control-versus-ablation comparison chart.
- Positive-empty and empty-FP trade-off chart.
- Patient Dice heatmap and V104/V116 localization panels.
- A machine-readable go/no-go result naming the selected arm or the reason all arms failed.

## Remaining workflow

1. Mark 4C controlled ablation and validation gate.
2. Bounded continuation of the single winning configuration, only if its gate passes.
3. Freeze preprocessing, threshold, checkpoint, and inference policy.
4. One-time locked test evaluation after all validation gates pass.
5. Final patient/lesion-stratified analysis, limitations, reproducibility package, and research-paper reporting.

## Evidence

- `mark_4b_roi_probability_diagnostics.ipynb`
- `mark_4b_outputs/mark_4b_gate_result.json`
- `mark_4b_outputs/threshold_results.csv`
- `mark_4b_outputs/threshold_patient_metrics.csv`
- `mark_4b_outputs/bootstrap_confidence_intervals.csv`
- `mark_4b_outputs/calibration_frontier_dashboard.png`
- `mark_4b_outputs/localization_volume_104.png`
- `mark_4b_outputs/localization_volume_116.png`

## Mark 4C update — 2026-08-02

- **Overall workflow:** approximately **70% complete**.
- Both bounded ablation arms completed and the notebook now executes without errors.
- The two-channel arm collapsed to an all-background solution on tumour-positive patients and is rejected.
- The recall-loss arm achieved the intended recall improvement: positive predicted-empty fell from 36.85% to 30.13%, Q1 detection increased from 42.59% to 47.91%, and V104 Dice increased from 0.0672 to 0.1038.
- The recall-loss arm still failed V116: Dice fell from 0.01078 to 0.00085.
- A Mark 4C implementation inconsistency was identified: experimental-arm `mean_patient_dice` included four tumour-empty validation patients, while the reused control metric was calculated over nine tumour-positive patients. Recalculation over the same nine positive patients gives recall-loss mean Dice 0.3749 and two-channel mean Dice approximately zero.
- Corrected interpretation: recall-loss passes five of six continuation targets; V116 is the only remaining failed target. The saved gate JSON remains conservative (`fail`) but its experimental-arm mean-Dice values should not be used until a corrected evaluation notebook regenerates them.
- Next action: validation-only Mark 4D metric reconciliation and V116 lesion/localization diagnostic using saved checkpoints. Do not retrain and do not access the test split until that diagnostic determines a bounded intervention.

## Mark 4D update — 2026-08-02

- **Overall workflow:** approximately **73% complete**.
- Mark 4D completed without notebook errors and regenerated metrics using the same nine tumour-positive validation patients for both checkpoints.
- No checkpoint/threshold pair passed all six continuation targets.
- Recall-loss at threshold 0.60 passed five targets: positive-patient mean Dice 0.3766, V104 Dice 0.1002, Q1 detection 47.91%, positive predicted-empty 30.52%, and empty-slice FP 4.79%. V116 Dice remained failed at 0.00071.
- Control retained V116 Dice above target through threshold 0.65 but failed positive predicted-empty at every threshold.
- V116 failure is not ROI clipping: 100% of its 152,763 tumour pixels are inside the frozen ROI.
- V116 failure is not limited to small lesions. Median truth-region probability is zero in every V116 lesion-size quartile; only 3.0% of V116 positive slices are detected by recall-loss at threshold 0.50.
- Next action: Mark 4E validation-only checkpoint-fusion diagnostic. Evaluate pixelwise maximum, mean, and bounded weighted probability fusion across thresholds using the existing control and recall-loss caches. Retrain only if no fusion policy passes the complete gate. Test remains locked.

## Mark 4E update — 2026-08-03

- **Overall workflow:** approximately **77% complete**.
- Mark 4E executed without errors and evaluated seven frozen fusion policies across fourteen thresholds using the corrected nine-positive-patient definition.
- Pixelwise maximum fusion at threshold 0.70 passed all six temporary continuation targets: mean positive-patient Dice 0.3771, V104 Dice 0.1166, V116 Dice 0.01047, Q1 detection 50.57%, positive predicted-empty 27.45%, and empty-slice FP 5.55%.
- Mean, 75% control/25% recall, and 25% control/75% recall fusion also passed. Maximum fusion was selected by the predeclared highest-mean-Dice rule.
- The V116 margin remains narrow, so this is a temporary validation pass rather than evidence of robust generalization or permission to use the test split.
- Ten major project steps are complete through Mark 4E.
- Before further training, complete four train/validation-only dataset-characterization tasks: geometry/acquisition profiling, lesion morphology and burden profiling, HU/contrast/domain-shift profiling with V116 analog search, and difficulty/leakage/QC review. Freeze the resulting data card and sampling policy.
- After the dataset audit, remaining model/evaluation work is: frozen-fusion confirmation, at most one evidence-driven bounded training refinement if needed, final inference-policy freeze, one-time locked test evaluation, and final reporting.

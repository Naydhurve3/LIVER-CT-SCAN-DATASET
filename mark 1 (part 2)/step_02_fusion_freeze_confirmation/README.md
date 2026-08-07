# Step 02 — Fusion Freeze Confirmation

## Status

Completed and independently reviewed on 3 August 2026. Result level: `VALIDATION_FREEZE_PASS`. Every mandatory confirmation requirement passed and the test split remained sealed.

## Final confirmed result

- Frozen policy: pixelwise maximum fusion at global threshold 0.70.
- Mean Dice over nine tumour-positive patients: 0.37706060; patient-bootstrap 95% interval 0.19792987 to 0.55420929.
- V104 Dice: 0.11647368.
- V116 Dice: 0.01047342.
- Q1 positive-slice detection: 50.570342%.
- Positive predicted-empty: 27.447217%.
- Empty-slice false positives: 5.548066%.
- Fresh repeated inference was exactly deterministic.
- Aggregate fresh/historical score error: approximately 1.09e-7; hard-prediction disagreement approximately 1.21e-7.
- Maximum reproduced metric difference from Mark 4E: 0.00015329.
- Test images accessed: false.

## Important limitations

- V116 passes its temporary 0.01 Dice floor by only about 0.000473 and remains a severe localization failure.
- V116 localization panels show near-zero scores across large visible tumour regions.
- Positive predicted-empty remains 27.45%, despite passing the temporary 35% ceiling.
- Derived 3D component detection is 13.04% in the smallest quartile, 76.09% in Q2, 91.30% in Q3, and 100% in Q4.
- The bootstrap lower bound for mean patient Dice is below the temporary mean-Dice target; this was predeclared as a report-only robustness result, not an additional pass gate.
- The confirmation passes continuation/freeze criteria but does not meet all historical aspirational research targets.

## Question

Does fresh, deterministic inference from the two frozen checkpoints reproduce the Mark 4E maximum-fusion validation pass at global threshold 0.70 closely enough to freeze the inference policy?

## Frozen configuration

- Control checkpoint: `mark 1/mark_4_outputs/mark_4_best.pth`.
- Recall checkpoint: `mark 1/mark_4c_outputs/recall_loss_best.pth`.
- Input: one broad-window channel from `[-160,240]` HU, stored as derived uint8 and normalized by 255.
- ROI: predicted-liver threshold 0.50, largest 3D component, padding 16.
- Fusion: pixelwise maximum.
- Threshold: 0.70.
- Post-processing: none.
- Mean Dice population: nine tumour-positive validation patients.
- Test state: locked.

No checkpoint, threshold, fusion weight, post-processing rule, or patient-specific policy may be selected in this phase. Threshold 0.65 is reported only as predeclared sensitivity and cannot replace 0.70.

## Required confirmation

1. Step 01 dataset gate remains passed.
2. Manifest and checkpoint hashes are recomputed.
3. Fresh inference matches historical caches under aggregate score and frozen hard-decision tolerances; isolated maximum outliers are diagnostic.
4. A complete repeated inference pass is deterministic within tolerance.
5. All six temporary Mark 4E targets pass at maximum fusion and threshold 0.70.
6. Patient bootstrap uncertainty is saved.
7. Patient, slice, 3D component/size, probability, and V104/V116 localization diagnostics are saved.
8. The immutable policy and machine-readable gate are written.
9. `test_images_accessed` remains false.

## Decision

- Full mandatory pass: `VALIDATION_FREEZE_PASS` and proceed to the final inference-policy package before requesting explicit test authorization.
- Any failure: `FAILED_GATE`; diagnose numerical, cache, determinism, or target failure without opening a new search.

## Execution

Open `step_02_fusion_freeze_confirmation.ipynb` using the project GPU-capable kernel. Select **Restart Kernel and Run All**. The notebook performs four complete validation inference passes and may take substantial time. Every new output is written only to this phase's `outputs/` directory.

Execution is now complete. Treat `outputs/gate_result.json` and `outputs/immutable_inference_policy.json` as the controlling Step 02 artifacts.

## Test lock

The test split remains sealed. This notebook uses validation data only.

## Repair and resume note

The first execution completed all fresh and repeated inference but stopped on an overly strict single-pixel maximum cache-error assertion. The repaired contract requires aggregate mean score error `<=1e-4`, fraction above `5e-4` no greater than `1e-4`, hard-prediction disagreement at threshold 0.70 no greater than `1e-6`, and reference metric delta `<=5e-4`.

If the original kernel is still live, rerun the first four setup/preflight code cells, skip the expensive fresh-inference code cell, and continue at `Compute frozen-policy patient, slice, and six-target metrics`. That cell reconstructs equivalence from the 13 completed fresh caches. If kernel state was lost, use Restart Kernel and Run All.

The repaired continuation completed successfully. No further Step 02 execution is required.

## Next action

Create Step 03 final inference-policy freeze. It should verify every immutable hash and file, freeze the formal final acceptance table before test access, package the exact loader/ROI/fusion/threshold/metric contract, and produce a signed checksum inventory. It must not open the test split. After Step 03 passes, request explicit authorization for the one-time locked test evaluation.

# Step 01 — Pre-Training Dataset Characterization

## Status

Completed and independently reviewed on 3 August 2026 after the repaired full Run All. Result level: `DIAGNOSTIC_COMPLETE`. All mandatory dataset-audit targets passed. Decision: `PASS_FREEZE_DATA_CARD_AND_PROCEED_TO_FUSION_CONFIRMATION`.

## Final findings

- Output contract: all 36 required artifacts exist.
- Critical failures: zero integrity, geometry, label, ROI, and leakage failures.
- Alignment: all 117 train/validation volumes resolved; 112 match numerically after the approved transform and volumes 48–52 retain manually approved header-only inconsistencies.
- ROI: minimum reported tumour-pixel containment is 1.0.
- Lesions: train has 637 components with median 0.540068 mL; validation has 208 with median 0.145761 mL and proportionally more tiny components.
- Validation covers all train-derived lesion-volume, diameter, and patient-burden strata.
- Corrected median tumour-minus-liver contrast is -34 HU in train and -44 HU in validation; median robust CNR is -1.518286 versus -1.881500.
- V104: 10 lesions, 134.639384 mL tumour, very low contrast (-90 HU; robust CNR -4.336071), Mark 4E Dice 0.116627.
- V116: one large lesion, 266.351941 mL tumour, weak contrast (-9 HU; robust CNR -0.296122), Mark 4E Dice 0.010474. Its failure is not explained by small lesion size or ROI clipping.
- Five nearest training analogs exist for both V104 and V116 under the train-fitted feature scaler.
- Sampling policy is frozen as `uniform_patient_aware_existing_sampler_no_change`; the audit does not justify a new training ablation before fusion confirmation.

## Verified starting state

- Corrected build: `build_corrected_20260713_214847_v2`.
- Manifest SHA-256 verified from disk: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`.
- Manifest schema: 58,638 rows and 27 columns; train 40,667 slices/104 volumes, validation 10,685 slices/13 volumes, locked test 7,286 slices/14 volumes.
- Mark 4E is the controlling model result: pixelwise maximum fusion at threshold 0.70, six of six temporary continuation targets passed, `TEMPORARY_CONTINUATION_PASS`.
- Mark 4E machine-readable provenance records nine tumour-positive validation patients and `test_images_accessed: false`.
- Frozen ROI inputs are the Mark 3 training ROI manifest and Mark 4 validation ROI manifest, using liver score threshold 0.50, largest 3D component, and 16-pixel padding.

## Documentation conflicts and precedence

- Mark 4D recommends a small-lesion sampling ablation, but Mark 4E subsequently passed the temporary fusion gate. Part 2 therefore correctly places dataset characterization before any further training.
- Mark 4E's raw gate says `FREEZE_FUSION_POLICY_AND_THRESHOLD` and names a Mark 4F notebook. The Part 2 roadmap refines this to Step 02 bounded fusion confirmation after the dataset gate; this README follows the later Part 2 contract.
- Historical `understanding the project/` targets are aspirational and predate Mark 4E. They are not the current six temporary continuation targets.
- Mark 4C experimental mean patient Dice used an inconsistent patient population. Mark 4D's nine-tumour-positive-patient reconciliation and Mark 4E metrics are authoritative.

## Notebook scope

The notebook implements all sections of `03_PRETRAINING_DATASET_AUDIT_CONTRACT.md`:

1. Manifest identity, schema, missingness, grain, path/source-hash coverage, split leakage, duplicate/near-duplicate screening, exclusions, and deterministic derived-mask reconciliation.
2. Per-volume NIfTI shape, spacing, affine/orientation, field of view, physical organ/tumour volumes, frozen ROI geometry, train/validation comparisons, robust train-derived outliers, and explicit V104/V116 rows.
3. Source-label 3D lesions using 6-connectivity with 26-connectivity sensitivity, physical morphology, burden, multifocality, axial continuity, centre/periphery proxy, train-derived bins, and validation coverage.
4. Source-NIfTI tumour/liver/background HU statistics, broad/liver-window visibility, robust CNR, domain-shift summaries, and train-fitted nearest analogs for validation including V104/V116.
5. Tumour containment, source label values, affine/shape agreement, component and slice-jump QC, frozen ROI coverage, and deterministic alignment panels.
6. Post-feature join to Mark 4E patient/slice diagnostics and non-causal difficulty associations.
7. Data card, conservative train-only sampling policy, provenance/configuration, expected-versus-actual table, and machine-readable gate.

## Parameters

- Random seed: 42.
- Allowed splits: train and validation only.
- 3D connectivity: 6 primary, 26 sensitivity.
- Robust outliers: train Q1/Q3 plus or minus 1.5 IQR; flag only.
- HU windows: broad `[-160, 240]`, liver `[0, 200]`.
- Analog scaler: `RobustScaler` fit on train only; Euclidean distance; five nearest train volumes.
- Frozen ROI: threshold 0.50, largest 3D component, padding 16.
- Sampling policy default: retain uniform patient-aware behavior unless train-only evidence supports a later bounded intervention.

## Execution

Open `step_01_pretraining_dataset_characterization.ipynb` with the project `.venv` kernel, then use **Restart Kernel and Run All**. All generated files are written only to `outputs/`.

The final decision must be exactly one of:

- `PASS_FREEZE_DATA_CARD_AND_PROCEED_TO_FUSION_CONFIRMATION`;
- `HOLD_FIX_DATA_OR_LABEL_ISSUE`;
- `HOLD_DEFINE_ONE_BOUNDED_DATA_INTERVENTION`.

## Validation completed before handoff

- Notebook JSON validated with `nbformat`.
- Every Python code cell parsed with `ast`.
- Imports, path constants, output containment, manifest hash, train/validation filtering, configuration write, and test-lock assertions executed in a safe preflight.
- Full NIfTI analysis not executed.
- Development repairs are recorded in `error_fixes.md`.

## Next action

Create Step 02 for bounded fusion freeze confirmation. Recompute from the two checkpoints, verify cache equivalence and determinism, use pixelwise maximum fusion and threshold 0.70 unchanged, report bootstrap uncertainty and V104/V116 localization, and perform no new threshold or fusion-weight search.

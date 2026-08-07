# Step 03 — Final Inference-Policy Freeze

## Purpose

This phase packages the completed Step 01 dataset gate and Step 02 validation-freeze pass into one immutable final inference contract. It performs no training or inference and never opens the test split.

The notebook verifies source hashes, checksums the full frozen artifact chain, freezes the exact loader/ROI/fusion/threshold/metric rules, and predeclares the formal minimum final acceptance table before any test access.

## Verified completion — 3 August 2026

- Result level: `VALIDATION_FREEZE_PASS`.
- Mandatory readiness requirements: 10/10 passed.
- Frozen artifacts checksummed: 38, including all 13 validation probability caches and the predicted-liver ROI generator checkpoint.
- Step 01 and Step 02 gates, manifest, checkpoints, validation ROI manifest, loader source, model source, metrics, notebook, and cache artifacts are included in the checksum chain.
- Every Step 02/Step 03 immutable policy comparison passed.
- Frozen policy: pixelwise maximum fusion, global threshold 0.70, and no post-processing.
- Frozen ROI generation: checkpoint hash `9c4160bbd68891f9dc4e5f04ceca4391f38c5869b3f81c72b95d4639e0572223`, liver output channel 0, robust per-slice normalization, threshold 0.50, largest 26-connected 3D component, padding 16, and full-image fallback for an empty ROI.
- Frozen Q1 test definition: positive slices with 1–51 tumour pixels; 51 is the training-only 25th percentile across 4,930 positive training slices.
- Formal minimum acceptance is declared in `outputs/final_acceptance_contract.json`; historical stronger targets remain non-mandatory research aspirations.
- Authorization readiness: `READY_TO_REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION`.
- Authorization granted: false.
- Test images accessed: false.
- Notebook validation: `nbformat` valid, all seven code cells parse, and the complete lightweight run succeeded.

## Run

Open `step_03_final_inference_policy_freeze.ipynb` with the project Python kernel and use **Restart Kernel and Run All**. The run is lightweight: it reads prerequisite artifacts and checkpoint bytes only for integrity hashing. It does not instantiate a dataset loader or model.

All generated artifacts are written only to `outputs/`.

## Decision boundary

A full pass retains result level `VALIDATION_FREEZE_PASS` and changes readiness to `READY_TO_REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION`. This is not test authorization and is not final model success.

The historical stronger performance targets remain research aspirations. V104 and V116 remain validation-only hard-case sentinels; they are not relabelled as test patients.

## Test lock

`test_images_accessed`, `test_masks_accessed`, `test_probabilities_accessed`, and `test_statistics_computed` must all remain false. The next phase may be created or run only after the project owner explicitly authorizes exactly one locked test evaluation using the frozen policy and acceptance contract.

## Next action after a passing run

Review `outputs/gate_result.json`, `outputs/final_inference_policy.json`, and `outputs/final_acceptance_contract.json`, then request explicit one-time test authorization. Do not inspect test data before that authorization is recorded.

No Step 04 notebook has been created or run. The exact authorization sentence is stored in `outputs/authorization_readiness.json`.

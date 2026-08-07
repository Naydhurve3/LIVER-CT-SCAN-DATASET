# Step 04 — One-Time Locked Test Evaluation

## Status

Notebook created and safely preflighted. One-time owner authorization was recorded on 5 August 2026, but the test split has not yet been opened by the agent.

## Frozen evaluation

- Predicted-liver ROI: frozen two-output liver checkpoint, channel 0, robust per-slice normalization, threshold 0.50, largest 26-connected 3D component, padding 16, and full-image fallback for an empty ROI.
- Tumour checkpoints: frozen control and recall-loss checkpoints.
- Fusion: pixelwise maximum.
- Global hard threshold: 0.70.
- Post-processing: none.
- Q1 definition: positive slices with 1–51 tumour pixels, with 51 derived from training only.
- Bootstrap: 10,000 tumour-positive patient resamples, seed 42.

## Authorization boundary

The owner instructed the agent to make the required authorization edit after the locked assertion stopped execution. The notebook and generator now record:

- `AUTHORIZATION_GRANTED = True`
- Authorization UTC: `2026-08-05T11:39:37.3247843Z`
- One-time run UUID: `871d289b-bf6b-4346-978f-2df02ade26ab`
- Canonical authorization text: “I explicitly authorize one-time locked test evaluation using the Step 03 frozen policy and acceptance contract.”

Use **Restart Kernel and Run All exactly once**. Do not regenerate the notebook first because that would discard execution outputs.

## Outputs after the authorized run

All results will be written only under `outputs/`, including full probability caches, patient/slice/global metrics, bootstrap uncertainty, train-derived lesion-size metrics, probability diagnostics, failure panels, signed provenance, acceptance results, and the sealed run ledger.

## Current next action

Restart the kernel and Run All exactly once. If execution is interrupted, preserve the same UUID and follow the resume instructions rather than starting another test run.

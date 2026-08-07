# Step 1 — Smoke Gate (16-slice overfit, four candidates)

## Question
Can each improvement candidate (Control, C1 high-res ROI, C2 analog sampler, C3
capped-recall loss) overfit a deterministic 16-slice dev set to hard micro-Dice
>= 0.80? This is a trainability canary, not a selection gate.

## Inputs
- Corrected LiTS manifest (hash must match `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`)
- Phase 0 sealed `EVAL_VOLUMES.json` (8-volume holdout excluded here)
- Phase 0 `sampling.json` (C2 analog policy)
- `mark 1/mark_3_outputs/training_roi_manifest.csv` (per-volume ROI boxes for C1)
- Framework: `VerifiedManifestDataset`, `MobileNetV2UNet`, `FocalDiceLoss`, `StabilityBoundedRecallLoss`

## Parameters
- Seed 42; deterministic 16-slice overfit (8 tumour-positive + 8 negative, shuffled)
- Preprocessing frozen: HU [-160,240], bilinear image / nearest mask to 256x256
- C1 only: crop per-volume predicted-liver ROI box, up-sample to 256
- Cold-start arms (ImageNet encoder, random tail); GPU only
- Control/C1/C2: FocalDiceLoss; C3: StabilityBoundedRecallLoss (cap 2.0)
- AdamW lr 3e-4 (C3 1e-4), cosine, gradient clip 5.0, ~60 epochs over 16 slices
- Gate: hard micro-Dice >= 0.80 for EVERY arm

## Execution
1. `python create_step_1_smoke.py` to (re)build the notebook.
2. Run `step_1_smoke.ipynb` end to end on GPU.

## Decision gate (`outputs/gate_result.json`)
- All arms >= 0.80  -> `PROCEED_TO_STEP_2_TRAIN`
- Any arm < 0.80     -> `HALT_REVIEW_ARM`

## Outputs (`outputs/`)
- `smoke_arm_results.csv` — per-arm hard micro-Dice + loss name
- `gate_result.json` — machine-readable decision, `test_images_accessed: false`

## Status
Generator built and validated (16 cells, all parse); preflight of dev split,
overfit selection, C1 ROI crop, both loss forwards passed. Heavy GPU training
not run autonomously.
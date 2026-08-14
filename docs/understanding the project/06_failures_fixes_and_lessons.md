# Failures, Fixes, and Lessons

## Dataset and scientific failures

### Invalid pairing/orientation

Symptom:

- Image and mask overlays were spatially inconsistent despite matching filenames and smooth axial continuity.

Cause:

- In-plane source orientation differed for a block of volumes.

Fix:

- Source-to-derived regeneration, zero-offset search, orientation scoring, manual spatial approval, corrected v2 build, manifest hashes.

Lesson:

- File counts, array shapes, and 3D continuity do not prove correct spatial pairing.

### Incorrect tumor-burden denominator

Symptom:

- Earlier tumor burden was under-reported.

Cause:

- A hard-coded `512×512` denominator was applied to `256×256` masks.

Fix:

- Use native mask height × width.

Lesson:

- Metric denominators must come from the evaluated artifact, not assumed source dimensions.

### Mask semantic ambiguity

Symptom:

- Older documentation implied liver-plus-tumor foreground while modeling used tumor-only targets.

Fix:

- Lock tumor-only model target and keep organ masks separate.

Lesson:

- Never compare results until target semantics are explicit and identical.

## Runtime notebook failures and fixes

### Matplotlib boxplot `labels` error

Error:

`TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'`

Cause:

- Matplotlib version expected the newer argument name.

Fix:

- Use the compatible tick-label argument or set tick labels separately.

### RNG checkpoint type error

Error:

`TypeError: RNG state must be a torch.ByteTensor`

Cause:

- Restored serialized RNG state was not normalized to a CPU `uint8` tensor.

Fix:

- Apply `.detach().cpu().to(torch.uint8)` before restoring PyTorch, CUDA, and sampler-generator states.

### Missing GPU temperature constant

Error:

`NameError: MAX_GPU_TEMP_C is not defined`

Cause:

- Visualization referenced an older variable name.

Fix:

- Use the current thermal constants consistently and provide bounded defaults.

### Non-finite loss at epochs 7/8

Error:

`FloatingPointError: Non-finite loss`

Contributing risks:

- Mixed precision, unstable loss behavior, and gradient magnitude.

Fixes:

- Finite-loss checks.
- Gradient clipping at norm 5.
- Stable loss mixtures.
- Safe epoch checkpoints and exact resume.

### 2.5D review-channel mismatch

Error:

`expected input ... to have 3 channels, but got 1`

Cause:

- Training used the adjacent-slice dataset, but the review cell rebuilt a single-channel `VerifiedManifestDataset`.

Fix:

- Review with a `Subset` of the existing 3-channel validation dataset.
- Display the central channel at index 1.

### GPU thermal interruption

Symptom:

- Multi-task training stopped after epoch 1 when the next start exceeded 84°C.

Fix:

- Dedicated continuation notebook.
- Bounded 30-second cooling polls.
- Exact checkpoint and RNG restoration.
- Per-epoch thermal logs and progress dashboards.

Observed:

- Epoch-end temperatures `86–87°C`.
- Next-epoch starts typically cooled to `73–77°C`.
- No `90°C` emergency crossing.

## Modeling failures

### Aggregate Dice hid patient collapse

The baseline reached global Dice `0.5584`, but volume 116 remained at zero and worst-patient Dice was effectively zero.

Lesson:

- Patient-level, size-stratified, and empty-prediction metrics are mandatory.

### Balanced sampling improved Q1 but hurt the mean

Q1 detection improved to `53.99%`, but mean patient Dice fell to `0.3562`.

Lesson:

- Sampling can redistribute performance rather than improve it.

### Pure recall-aware Tversky collapsed

It produced 100% positive-slice empty predictions.

Lesson:

- A theoretically recall-oriented loss can still converge to a degenerate solution under extreme imbalance and implementation/optimization conditions.

### Composite loss recovered one patient but harmed the cohort

Volume 116 reached `0.0515`, but mean patient Dice fell to `0.3229`.

Lesson:

- Do not select a model because one difficult patient improves.

### Organ-assisted normalization helped but leaked inference information

It produced the strongest mean patient Dice and volume 104 result, but it used the ground-truth organ mask to compute validation normalization statistics.

Lesson:

- This run is useful diagnostic evidence, not a deployable performance claim.

### 2.5D context suppressed too much

Empty-slice FP fell to `2.37%`, but Q1 detection fell to `25.48%` and key patients were missed.

Lesson:

- More spatial context does not automatically improve small-lesion or patient generalization.

### 3D cleanup could not solve a probability problem

Every hysteresis/component rule lost patient Dice before false positives reached target.

Lesson:

- Post-processing cannot create missing tumor signal.

### Multi-task gating was a no-op

Raw and liver-gated predictions were numerically identical at the best epoch.

Cause:

- Every tumor prediction already lay inside the large dilated predicted-liver support.

Lesson:

- The remaining errors are intra-liver discrimination errors. Changing only the binary liver gate is unlikely to help.

### Multi-task model over-suppressed tumor

From epoch 1 to epoch 8:

- Empty-slice FP improved from `15.64%` to `3.25%`.
- Positive predicted-empty worsened from `20.63%` to `46.16%`.
- Volumes 104 and 116 fell to approximately zero.

Lesson:

- The next investigation should calibrate probabilities and explicitly measure false-negative/false-positive tradeoffs before changing architecture.

## Practices to retain

- Manifest and checkpoint hashes.
- Patient-disjoint validation.
- Test locking.
- Deterministic seeds.
- Per-epoch checkpoint and CSV histories.
- Exact RNG resume.
- Thermal cooldown.
- Expected-versus-actual dashboards.
- Patient and lesion-size tables.
- Explicit pass/fail gate JSON.


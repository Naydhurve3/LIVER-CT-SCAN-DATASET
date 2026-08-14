# Parameters and Techniques

## Shared reproducibility settings

- Random seed: `42`.
- Framework: PyTorch.
- Primary architecture: MobileNetV2-U-Net.
- Pretrained encoder: `False` in the recorded controlled gates.
- Weight decay: generally `1e-4`.
- Mixed precision: enabled on CUDA where supported.
- Deterministic cuDNN: enabled.
- Benchmark mode: disabled.
- Gradient clipping: maximum norm `5.0` in stabilized continuations.
- Validation augmentation: none.
- Test loader: locked.

## Base model

Single-slice experiments:

- Input: `[batch, 1, 256, 256]`.
- Output: `[batch, 1, 256, 256]`.
- Target: binary tumor mask.

2.5D experiment:

- Input: `[batch, 3, 256, 256]`.
- Channels: previous, current, next slice.
- Output: one tumor channel.

Multi-task experiment:

- Input: `[batch, 1, 256, 256]`.
- Output channel 0: liver.
- Output channel 1: tumor.
- Tumor output head warm-started from the single-output intensity checkpoint.

## Training parameters by main stage

| Stage | Epoch plan | Batch | Val batch | LR | Min LR | Positive weight |
|---|---:|---:|---:|---:|---:|---:|
| 16-slice overfit | max 150 | 4 | — | `1e-3` | — | deterministic subset |
| 5-epoch smoke | 5 | 8 | 16 | `3e-4` | — | 4 |
| Patient-aware baseline | 25 plan; decision at 10 | 8 | 16 | `3e-4` | `3e-6` | 3 |
| Intensity robustness | 25 plan; decision at 10 | 8 | 16 | `3e-4` | `3e-6` | 3 |
| Multi-task | 15 plan; decision at 10 | 8 | 16 | `1e-4` | `2e-6` | 3 |

Schedulers used cosine annealing. Optimizer was AdamW.

## Losses evaluated

### Focal-Dice

The strongest stable general baseline. Typical configuration:

- Focal alpha `0.75`.
- Focal gamma `2.0`.
- Focal weight `0.5`.
- Dice weight `0.5`.

### Focal Tversky

- Alpha `0.40`.
- Beta `0.60`.
- Gamma `0.75`.

This pure recall-aware experiment collapsed to empty predictions.

### Stabilized composite

- Focal-Dice contribution `0.75`.
- Focal-Tversky contribution `0.25`.

This prevented complete collapse but still reduced mean patient Dice.

### Multi-task loss

- Tumor loss weight `0.72`.
- Liver loss weight `0.23`.
- Tumor-outside-liver containment weight `0.05`.
- Liver loss combined BCE and soft Dice.
- Tumor loss used Focal-Dice.

## Sampling strategies

### Positive-slice weighted sampling

Baseline long-run weight: `3`.

Smoke-test weight: `4`.

Purpose: counter the low proportion of tumor-positive slices.

Risk observed: excessive positive weighting can increase tumor predictions on tumor-negative liver slices.

### Inverse-volume × lesion-quartile sampling

Purpose:

- Prevent large patients from dominating.
- Increase small-lesion exposure.

Outcome:

- Q1 detection improved.
- Mean patient Dice and key-patient results worsened.

## Preprocessing strategies

### Corrected build preprocessing

- Source HU window recorded as `[-160, 240]`.
- Resize to `256 × 256`.
- Bilinear image resizing.
- Nearest-neighbor mask resizing.
- Orientation correction after mask derivation/resizing according to the build profile.

### Organ-assisted robust normalization

- Reference pixels taken from ground-truth liver/organ mask.
- Center: median.
- Scale: IQR / `1.349`.
- Clip: ±3 robust standard deviations.
- Rescale to `[0,1]`.

This was diagnostically strong but is not valid as a deployable validation/test preprocessing step when ground-truth organ masks are unavailable.

### Image-only robust normalization

- Reference: non-zero image pixels.
- Median center.
- IQR / `1.349` scale.
- Clip ±3.
- Rescale to `[0,1]`.

Used by the multi-task experiment to remove ground-truth-organ dependence.

### Augmentation

Common paired geometric augmentation:

- Horizontal flip probability `0.30`.
- Affine probability `0.60`.
- Rotation up to ±10 degrees.
- Translation up to 5% of width/height.
- Scale `0.95–1.05`.

Intensity robustness augmentation:

- Gamma range `0.85–1.15`, probability `0.50`.
- Gaussian noise standard deviation `0–0.025`, probability `0.35`.

## Predicted-liver gating

- Tumor threshold: `0.50`.
- Liver threshold: `0.50`.
- Liver dilation/max-pooling kernel: `31`.

Observed result: raw and gated tumor predictions were identical at the best epoch. The predicted-liver support contained every tumor prediction, so this gating configuration was ineffective.

## 3D post-processing grid

Evaluated:

- Raw thresholds `0.20`, `0.30`, `0.50`.
- Low/high hysteresis combinations.
- Minimum components of 16 or 64 voxels.
- Minimum axial span of 1 or 2 slices.
- Six-connected 3D component structure.

Best configuration: unchanged raw threshold `0.50`.

## Thermal and interruption controls

- Desired start temperature: below `84°C`.
- Emergency limit: `90°C`.
- Cooldown polling: every 30 seconds.
- Maximum bounded cooldown: 30 minutes.
- Checkpoint saved after every completed epoch.
- Continuation restores Python, NumPy, PyTorch, CUDA, and sampler RNG states.

During multi-task continuation:

- Epoch-end GPU temperatures were commonly `86–87°C`.
- Cooling checks typically brought the next start to `73–77°C`.
- Emergency threshold was not crossed.


# FAUP-Net: Uncertainty-Gated Skip Connections

## Scope (Per Blueprint ADR-006)
**Scoped implementation only** — gated skip connections + UWACL-v2.
Full multi-level encoder/decoder uncertainty propagation is deferred to future work.

## Problem
Standard U-Net skip connections blindly forward encoder features to the decoder,
including features from uncertain or corrupted regions. This propagates noise and
leads to false positives in segmentation.

## Approach
Gate each encoder-to-decoder skip connection by a learned uncertainty estimate:

1. **Uncertainty Estimation**: A mini-ensemble head on each encoder block produces
   a pixel-wise variance map via Monte Carlo dropout or feature statistics.
2. **Soft Gating**: Multiply the skip feature map by `(1 - σ²)`, where σ² is the
   normalized uncertainty map. High-uncertainty regions are suppressed.
3. **Feature Fusion**: Gated skip features are concatenated with decoder features
   (standard U-Net) before the next decoder convolution.

## Mathematical Formulation

Let `e_i` be the encoder feature map at level `i`, and `u_i` the estimated
uncertainty map. The gated skip is:

    g_i = e_i ⊙ (1 - σ²_normalized(u_i))
    d_i = Conv(Concat(upsample(d_{i+1}), g_i))

where σ²_normalized normalizes uncertainty to [0,1].

## Uncertainty Head Design
- Applied to the **last 2 encoder blocks only** (not all 5 — VRAM optimization)
- Each head: Conv(3×3, out=1) + ReLU + dropout(p=0.1) — adds < 50K params
- Inference: single forward pass (not Monte Carlo); head output = uncertainty

## Expected Impact
- Suppresses uncertain features before decoder fusion
- Improves boundary definition where heterogeneity is high
- Estimated Dice improvement: +0.5–1.5% over baseline MobileNetV2U-Net

## Non-Goals (Explicitly Out of Scope)
- Multi-level encoder uncertainty propagation (all encoder levels)
- Decoder-side uncertainty propagation
- Uncertainty-guided decoding path selection
- 3D architectures — 2D slice-wise only

## Fallback Plan
If FAUP-Net Dice ≤ UP³RE-Net Dice:
- Publish "UWACL-v2 + Calibration Analysis" as Paper 1 at IEEE JBHI or Sci Reports
- FAUP-Net design and limited results go to supplementary material
- Full FAUP-Net exploration deferred to future work

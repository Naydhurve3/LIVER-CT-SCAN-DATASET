# UWACL-v2: Multi-Scale Uncertainty-Aware Combined Loss

## Motivation
UWACL-v1 weights the per-pixel BCE+Dice loss by uncertainty, suppressing
high-uncertainty regions during training. UWACL-v2 extends this with:
1. **Multi-scale uncertainty aggregation** — capture uncertainty at multiple
   receptive field scales
2. **Edge-awareness** — explicitly penalize boundary errors via edge loss

## v1 Recap
UWACL-v1 computes per-pixel loss as:
    L = BCE * w + Dice * (1-w)
Weighted by: `1 + β * (1 - exp(-U / τ))`
where U is the per-pixel uncertainty map, β controls suppression strength,
and τ (tau) controls the sensitivity threshold.

## v2 Improvements

### 1. Multi-Scale Uncertainty
Instead of a single uncertainty map, aggregate uncertainty at 3 scales:
- **Fine** (1×1): per-pixel variance
- **Medium** (3×3): local neighborhood variance (avg pooled)
- **Coarse** (7×7): regional variance

Combined uncertainty: `U_multi = α_f * U_fine + α_m * U_medium + α_c * U_coarse`

### 2. Edge Loss
Explicit boundary supervision using high-pass filtering:
    L_edge = MSE(∇pred, ∇target)
where ∇ is a simple 3×3 Laplacian-like filter (difference from local average).

Added to total loss: `L_total = L_uwacl + λ * L_edge`

### 3. Tau Schedule Refinement
Linear warmup decay:
    τ_epoch = τ_initial * (1 - epoch/max_epochs) + τ_min
ensures the loss focuses on high-uncertainty regions early and gradually
becomes more permissive.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| beta | 5.0 | Uncertainty suppression strength |
| tau | 0.1 | Initial sensitivity threshold |
| tau_min | 0.01 | Minimum tau after schedule |
| tau_schedule | linear_decay | How tau evolves |
| edge_weight | 0.1 | Edge loss coefficient |
| scale_weights | [0.4, 0.35, 0.25] | Fine/medium/coarse uncertainty weights |

## Expected Impact
- Improved boundary accuracy (1-2% HD95 reduction)
- Better calibration (lower ECE)
- More robust to label noise in high-uncertainty regions

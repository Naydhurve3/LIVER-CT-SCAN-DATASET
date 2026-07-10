# UP³RE-Net: Research Novelty & Patent Analysis

**Title:** Uncertainty-Propagated Per-Pixel Reweighting Ensemble Network (UP³RE-Net)
**Task:** Binary liver tumor segmentation from 2D CT slices
**Hardware:** RTX 3050 Ti 4GB laptop GPU

---

## 1. Problem Statement

Liver tumor segmentation from CT scans faces two critical challenges:

1. **Extreme class imbalance**: 937:1 background-to-tumor ratio — standard losses under-perform
2. **High uncertainty regions**: tumor boundaries, small lesions, and ambiguous tissue — single models are overconfident at error locations

Existing solutions address these separately: ensemble methods improve accuracy but don't model uncertainty; uncertainty methods quantify confidence but don't use it to guide training. **No existing work combines both in a feedback loop.**

---

## 2. Novel Contribution

UP³RE-Net is a **two-stage ensemble with uncertainty feedback**:

- **Stage 1 (Bagging)**: Train 3 MobileNetV2U-Nets on stratified bootstrap samples
- **Stage 1.5 (Uncertainty)**: Compute per-pixel variance across ensemble predictions
- **Stage 2 (Boosting with UWACL)**: Train 4th model with loss weighted by per-pixel uncertainty
- **Inference**: Weighted ensemble of all 4 models + per-pixel confidence map

### Novel Components

| Component | Name | Novelty |
|-----------|------|---------|
| Per-pixel uncertainty feedback | — | Ensemble variance feeds back as pixel-wise loss weights |
| Loss function | **UWACL** | Uncertainty-Weighted Adaptive Compound Loss |
| Weight formula | `w = 1 + β·(1 - exp(-σ²/τ))` | Novel closed-form per-pixel weight from variance |
| Schedule | Dynamic τ decay | Uncertainty temperature decreases over epochs |
| Output format | Segmentation + confidence | Dual output for clinical interpretability |

---

## 3. Prior Art Analysis

### 3.1 Two-Layer Ensemble (Springer Cognitive Computation, 2024)

| Aspect | Their approach | UP³RE-Net |
|--------|---------------|-----------|
| Layer 1 | Train N models independently | Train 3 MNV2 on bootstraps |
| Layer 2 | Concatenate predictions as augmented input | Use **uncertainty** (variance) as **loss weight** |
| Weighting | Linear regression on validation | Per-pixel closed-form: `w = 1 + β·(1 - exp(-σ²/τ))` |

**Distinction**: They use predictions as **input features**; we use prediction variance as **loss weights**. This is fundamentally different: they augment the data representation, we modulate the optimization objective per-pixel.

### 3.2 Deep Ensemble with Novel Sampling + Loss (CompBioMed, 2024)

| Aspect | Their approach | UP³RE-Net |
|--------|---------------|-----------|
| Sampling | Slice-level class-balanced sampling | Bootstrap sampling (volume-level) |
| Loss | Exponential loss at pixel level | **Uncertainty-weighted** BCE + Dice |
| Ensemble | 2 UNet models averaged | 4 models with uncertainty-guided boosting |

**Distinction**: Their loss is static (exponential function of prediction error). Our loss is **dynamic**: weights come from an independent ensemble's variance, not from the training signal itself. This prevents feedback loops and provides an orthogonal signal.

### 3.3 UCTNet (Pattern Recognition, 2024)

| Aspect | Their approach | UP³RE-Net |
|--------|---------------|-----------|
| Uncertainty use | Route patches to CNN vs Transformer | Route pixel weight to loss function |
| Architecture | Single CNN-Transformer hybrid | Pure CNN ensemble |
| Task | Multi-organ segmentation | Binary tumor segmentation, 937:1 imbalance |

**Distinction**: They use uncertainty for **architectural routing** (model selection per patch). We use it for **optimization weighting** (loss modulation per pixel). Different mechanism, different purpose.

### 3.4 UG-CEMT (WACV 2025)

| Aspect | Their approach | UP³RE-Net |
|--------|---------------|-----------|
| Setting | Semi-supervised (10% labeled) | Fully supervised |
| Method | Mean teacher + cross-attention | Bagging + boosting + per-pixel re-weighting |
| Uncertainty | Consistency regularization | Ensemble variance → loss weight |

**Distinction**: Semi-supervised vs fully supervised. Different paradigm.

### 3.5 DyCON (2025)

| Aspect | Their approach | UP³RE-Net |
|--------|---------------|-----------|
| Setting | Semi-supervised | Fully supervised |
| Uncertainty | Dynamic weighting in consistency loss | Per-pixel weight from ensemble variance |
| Ensemble | Mean teacher (online + target) | Bagging + boosting (3 + 1) |

**Distinction**: Their uncertainty weights are from the model's own predictions (aleatoric). Our uncertainty weights are from **ensemble disagreement** (epistemic). These capture different types of uncertainty.

### 3.6 Key Gap Summary

**No existing work combines:**
1. Per-pixel uncertainty from ensemble variance
2. Fed back as per-pixel re-weighting weights into a boosting stage
3. With a dynamically scheduled uncertainty-weighted loss (UWACL)
4. For binary tumor segmentation with 937:1 class imbalance

This is the patent gap.

---

## 4. Patent Claims (Draft)

### Claim 1: Method for uncertainty-guided ensemble segmentation

A computer-implemented method for medical image segmentation, comprising:

(a) Training a plurality of base segmentation models on bootstrap samples of a training dataset, forming a bagging ensemble;

(b) Generating per-pixel uncertainty maps for at least a subset of the training dataset by computing variance of predictions across the bagging ensemble;

(c) Computing per-pixel weight maps from said uncertainty maps, wherein pixels with higher uncertainty receive higher weights;

(d) Training a boosting segmentation model on the training dataset using a loss function weighted by said per-pixel weight maps; and

(e) Combining predictions from the bagging ensemble and the boosting segmentation model to produce a final segmentation output.

### Claim 2: Per-pixel weight function

The method of claim 1, wherein the per-pixel weight map is computed as:

`w_ij = 1 + β * (1 - exp(-σ²_ij / τ))`

where:
- `σ²_ij` is the pixel-wise variance of predictions across the bagging ensemble at pixel (i,j);
- `β` is a scaling parameter controlling maximum weight amplification; and
- `τ` is a temperature parameter controlling sensitivity to variance.

### Claim 3: Uncertainty-weighted compound loss

The method of claim 1, wherein the loss function for training the boosting segmentation model comprises:

`L = (1/N) * Σ_ij [ w_ij * (λ₁ * BCE(p_ij, y_ij) + λ₂ * Dice(p_ij, y_ij)) ]`

where:
- `w_ij` is the per-pixel weight from claim 2;
- `BCE(p_ij, y_ij)` is binary cross-entropy loss at pixel (i,j);
- `Dice(p_ij, y_ij)` is Dice loss at pixel (i,j);
- `λ₁` and `λ₂` are component weighting parameters; and
- `N` is the total number of pixels.

### Claim 4: Dynamic threshold scheduling

The method of claim 2, wherein the temperature parameter τ is dynamically reduced over the course of training according to a schedule, such that:

- At early training stages, τ is set to a higher value, causing the weight map to primarily focus on high-uncertainty pixels; and
- At later training stages, τ is reduced toward zero, causing the weight map to approach uniform weighting across all pixels.

### Claim 5: Mutual consistency regularization

The method of claim 1, further comprising:

(f) Computing a consistency loss between predictions of the bagging ensemble members during training of the boosting segmentation model; and

(g) Using said consistency loss as an additional regularization term in the loss function.

### Claim 6: System for uncertainty-guided ensemble segmentation

A system for medical image segmentation, comprising:

(a) A memory storing a plurality of base segmentation models forming a bagging ensemble;

(b) A processor configured to:

   (i) Generate per-pixel uncertainty maps by computing variance of predictions across the bagging ensemble for input images;
   
   (ii) Compute per-pixel weight maps from said uncertainty maps;
   
   (iii) Execute a boosting segmentation model trained with a loss function weighted by said per-pixel weight maps; and
   
   (iv) Combine predictions from the bagging ensemble and the boosting segmentation model to produce a final segmentation output;

(c) An output interface configured to provide the final segmentation output and the per-pixel uncertainty maps.

---

## 5. Architecture Details

### 5.1 MobileNetV2U-Net

```
Input: (B, 1, 256, 256)
    │
    ▼
MobileNetV2 Encoder (torchvision, pretrained on ImageNet)
├── First conv: adapted from (3,32) → (1,32) — weight averaging
├── 7 inverted residual blocks → multi-scale features
└── Output: (B, 1280, 8, 8)
    │
    ▼
UNet Decoder (custom, from scratch)
├── 4 up-sampling blocks (bilinear + conv + skip connection)
├── Block 1: 1280 → 256 → 128
├── Block 2: 128 + skip → 64
├── Block 3: 64 + skip → 32
├── Block 4: 32 + skip → 16
└── Output conv: 16 → 1 (logits)

Output: (B, 1, 256, 256)
```

### 5.2 EnsembleWrapper

```
Input: (B, 1, 256, 256)
    │
    ▼
FOR each model in [m0, m1, m2, (m3)]:
├── Load model to GPU (swap if needed)
├── Forward pass: logits = model(x)
├── prob = sigmoid(logits)  // [B, 1, 256, 256]
├── Store pred = prob.cpu()
└── Unload model from GPU
    │
    ▼
mean_pred = mean(stacked_preds, dim=0)
variance  = var(stacked_preds, dim=0)
confidence = 1 - variance / max_var

Output: (mean_pred, variance, confidence)
```

### 5.3 UWACL Loss Computation

```
For a batch of B images at epoch e:

1. Load pre-computed uncertainty: σ² ∈ [B, 1, 256, 256]
2. Compute tau: τ_e = 0.1 * max(0.01, (1 - 0.9 * e / total_epochs))
3. Weight map: w = 1 + 5.0 * (1 - exp(-σ² / τ_e))
               w ∈ [1.0, ~5.88]
4. Compound loss:
   L_bce = F.binary_cross_entropy_with_logits(pred, target,
                                               pos_weight=10.0)
   L_dice = 1 - (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
   L_total = mean(w * (0.5 * L_bce + 0.5 * L_dice))
5. NaN guard: if isnan(L_total): skip batch
6. Backward: L_total.backward()
```

---

## 6. VRAM Management Strategy

| Phase | Max VRAM | Strategy |
|-------|----------|----------|
| Stage 1 training | ~3.2 GB | Single model (0.6 GB) + optimizer (1.2 GB) + activations (0.5 GB) |
| Uncertainty pre-compute | ~0.7 GB | Sequential model loading, predict, unload, next |
| Stage 2 training | ~3.3 GB | Same as Stage 1 + uncertainty map (~0.03 GB) |
| Ensemble inference | ~0.7 GB | Sequential model loading |

**Fallback**: If VRAM exceeds 3.5 GB at any point, skip current batch, call `torch.cuda.empty_cache()`, and log warning.

---

## 7. Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/losses.py` | ~80 | `DiceLoss`, `CombinedLoss`, `UncertaintyWeightedLoss` |
| `src/metrics.py` | ~60 | `dice_coefficient`, `iou_score`, `ensemble_uncertainty`, `calibration_error` |
| `src/models.py` | ~280 | `MobileNetV2UNet`, `EnsembleWrapper`, `create_model()`, `MODEL_REGISTRY` |
| `src/trainer.py` | ~220 | `Trainer` (Stage 1), `UncertaintyPrecomputer`, `UncertaintyGuidedTrainer` (Stage 2) |
| `src/config.py` | +15 | `PHASE4_RESEARCH_CONFIG` |
| `src/__init__.py` | +6 | New exports |
| `notebooks/04_research_training.ipynb` | ~300 | 8-cell research pipeline |

---

## 8. Research Outputs for Publication/Patent

| Output | Format | Content |
|--------|--------|---------|
| Comparison table | JSON + printed | Single vs Bagging vs UP³RE Full: Dice, IoU, ECE, params |
| Ablation table | JSON + printed | UWACL vs CombinedLoss: gains |
| Figure 1 | PNG | Bar chart — 4 methods × 3 metrics |
| Figure 2 | PNG | 6 test examples: image, GT, single pred, ensemble pred, uncertainty map |
| Figure 3 | PNG | Calibration plot: confidence vs accuracy |
| Figure 4 | PNG/Text | Architecture pipeline diagram |
| Patent draft | Markdown | 6 claims + background + description |
| Checkpoints | `.pth` | 4 trained models |

---

## 9. Prior Art References

1. Two-layer ensemble of deep learning models for medical image segmentation. *Cognitive Computation*, 2024. Springer.
2. A deep ensemble medical image segmentation with novel sampling method and loss function. *Computers in Biology and Medicine*, 2024.
3. UCTNet: Uncertainty-guided CNN-Transformer hybrid networks for medical image segmentation. *Pattern Recognition*, 2024.
4. Uncertainty-Guided Cross Attention Ensemble Mean Teacher for Semi-supervised Medical Image Segmentation. *WACV*, 2025.
5. DyCON: Dynamic Uncertainty-aware Consistency and Contrastive Learning. 2025.
6. Unified Focal loss: Generalising Dice and cross entropy-based losses. *Computerized Medical Imaging and Graphics*, 2022.
7. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS*, 2017. (Kendall & Gal — foundational uncertainty work)

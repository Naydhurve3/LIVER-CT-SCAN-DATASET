# Future Work & Research Directions

> **Liver Tumor Segmentation & Analysis Platform**
> Target: Publication-grade research extensions building on UP³RE-Net

---

## Table of Contents
1. [Quick-Win Ideas (Days)](#1-quick-win-ideas-days)
2. [Medium-Term Projects (1–2 Weeks)](#2-medium-term-projects-1-2-weeks)
3. [Major Architecture Overhaul (Weeks)](#3-major-architecture-overhaul-weeks)
4. [Research Paper Titles by Direction](#4-research-paper-titles-by-direction)
5. [Movable / Swappable Node Map](#5-movable--swappable-node-map)
6. [Full Ablation Study Design](#6-full-ablation-study-design)
7. [Target Venues by Ambition](#7-target-venues-by-ambition)

---

## 1. Quick-Win Ideas (Days)

These reuse existing trained models and dataset with minimal new engineering. No retraining required.

### 1.1 Uncertainty-Guided Human-AI Triage Study

**Concept:** Simulate a clinical workflow where the model flags the top-N% highest-uncertainty volumes for radiologist review. Measure manual review time saved vs. errors caught.

**Why it's novel:** Human-AI collaboration in diagnostic AI is a very active, fundable research direction. Per-pixel uncertainty from UP³RE-Net can power intelligent triage.

**What you already have:**
- Per-pixel uncertainty maps from UP³RE-Net ensemble variance
- Full test set predictions and ground truth
- Clinical analytics pipeline (tumor burden, patient profiles)

**What to measure:**
- AUC of uncertainty as an error detector (flag high-uncertainty slices → likely errors)
- Review burden reduction at various thresholds (flag 10%/20%/50% of slices)
- Trade-off curve: % of errors caught vs. % of slices requiring review
- Per-volume vs. per-slice triage comparison

**Estimated effort:** 1–2 days

---

### 1.2 Calibration Study & Reliability Diagrams

**Concept:** Take the existing ensemble, compute calibration curves, apply post-hoc temperature scaling or Platt scaling, compare before/after.

**Why it's novel:** "Trustworthy/calibrated medical AI" is one of the most cited topics in medical imaging venues. Reviewers love seeing calibration analysis even as a standalone contribution.

**What you already have:**
- Expected Calibration Error (ECE) in `src/metrics.py`
- Ensemble predictions with sigmoid probabilities
- Evaluation pipeline in `src/trainer.py:evaluate_model()`

**What to add:**
- Reliability diagrams (confidence vs. accuracy histograms)
- Per-class calibration plots
- Temperature scaling post-hoc optimization
- Brier score and MCE (Maximum Calibration Error)
- Comparison: single model vs. bagging vs. full UP³RE ensemble

**Estimated effort:** 1–2 days (no retraining)

---

### 1.3 LLM-Augmented Clinical Reporting

**Concept:** Feed segmentation + tumor burden analytics output into an LLM to auto-generate structured radiology-style reports.

**Why it's novel:** Multimodal AI + LLM-assisted clinical documentation is very trendy. You already have both halves (segmentation pipeline + LLM/RAG experience from CosmoGuide).

**What you already have:**
- Tumor burden analysis (volume, surface area, sphericity)
- Clinical insights engine (`src/analytics/clinical_insights.py`)
- Report generator (`src/analytics/report_generator.py`)

**What to add:**
- Structured prompt template for LLM
- Fields: tumor location, size class, morphology, risk level, uncertainty zones
- Output: natural-language radiology-style report paragraph
- Optionally: RAG with clinical knowledge base (`docs/clinical/`)

**Estimated effort:** 1–2 days

---

### 1.4 Knowledge Distillation for Edge Deployment

**Concept:** Distill the 3-model ensemble (teacher) into a single lightweight student model targeting near-ensemble accuracy at 1/3 inference cost.

**Why it's novel:** "Efficient AI / Edge AI for medical imaging" is extremely current. UP³RE-Net's ensemble is expensive (3–4 models); distilling to one is practically valuable.

**What you already have:**
- Trained ensemble members
- Training pipeline (`src/trainer.py`)
- MobileNetV2U-Net as student architecture

**What to add:**
- Distillation loss (KL divergence + soft labels from teacher)
- Temperature tuning for soft targets
- Compare: student vs. single model vs. ensemble on Dice, IoU, ECE
- Inference speed benchmark (ms per slice)

**Estimated effort:** 2–3 days

---

### 1.5 Low-Data Regime Study

**Concept:** Retrain (or fine-tune) with 10%, 25%, 50%, 100% of the 104 training volumes and plot the performance curve.

**Why it's novel:** "Data efficiency in medical imaging" (data is expensive to label) is a recurring, easy-to-motivate angle. Shows how much data your method actually needs.

**What you already have:**
- Full training pipeline with YAML configs
- Volume-wise splits in `data/splits/`
- Reproducible seed-based subsampling

**What to add:**
- Subsampling flag (random subset of TRAIN volumes)
- Learning curve plots: Dice vs. % training data
- Plateau detection: at what % data do returns diminish?
- Uncertainty quality vs. data volume analysis

**Estimated effort:** 2–3 days (pipeline reuses existing code)

---

## 2. Medium-Term Projects (1–2 Weeks)

### 2.1 Failure Mode & Interpretability Analysis

**Concept:** Systematic categorization of segmentation errors using Grad-CAM and error taxonomy.

**What to add:**
- Grad-CAM heatmaps on encoder features
- Error taxonomy: small lesion misses vs. boundary blur vs. false positives from artifacts
- Per-volume error profiles
- Correlation: high uncertainty regions → error types

**Estimated effort:** 3–5 days

---

### 2.2 Cross-Dataset Generalization Test

**Concept:** Evaluate the trained model on a different public liver dataset (e.g., 3D-IRCADb) without retraining to report out-of-distribution performance.

**Why it's novel:** "Does it generalize outside its training hospital?" — a major concern in medical AI.

**What to add:**
- Download and preprocess 3D-IRCADb or CHAOS dataset
- Adapt data loader for new naming convention
- Report: in-distribution (LiTS test) vs. OOD performance
- Uncertainty calibration shift analysis

**Estimated effort:** 3–5 days (no retraining, just eval)

---

### 2.3 Model Zoo Expansion

**Concept:** Add 3–5 baseline architectures for fair comparison.

**Models to add:**
| Model | Params | Notes |
|-------|--------|-------|
| U-Net (vanilla) | ~31M | Standard baseline |
| Attention U-Net | ~35M | Gate-based attention |
| U-Net++ | ~36M | Nested dense skip |
| DeepLabV3+ (MobileNetV2) | ~5.8M | Atrous conv baseline |
| TransUNet (light) | ~22M | Transformer encoder hybrid |

**Estimated effort:** 4–6 days (training all baselines)

---

## 3. Major Architecture Overhaul (Weeks)

### 3.1 FAUP-Net: Full-Architecture Uncertainty Propagation

The core novel contribution that extends UP³RE-Net from output-level uncertainty to full architectural uncertainty.

**Novel components:**

1. **FAUP Encoder Blocks** — Each encoder block outputs features + uncertainty map via a mini-ensemble (2 lightweight dropout heads). Standard: output only features. FAUP: output (features, uncertainty).
2. **Uncertainty-Gated Skip Connections** — Standard U-Net concats raw features. FAUP gates each skip connection by (1 - uncertainty), suppressing unreliable features before decoder fusion.
3. **FAUP Decoder Blocks** — Uncertainty-aware feature fusion: each decoder block weights its inputs (upsampled features + gated skip) by their combined reliability.
4. **Multi-Scale Disagreement Aggregator** — Ensemble disagreement is computed at every feature level (not just output), capturing where models disagree at different semantic scales.
5. **UWACL-v2 Loss** — Enhanced loss using multi-scale aggregated uncertainty instead of single-scale output variance, plus optional EdgeLoss for boundary awareness.

**Architecture diagram:**

```
Encoder                          Decoder
┌──────┐
│Block1│──▶ F1 + U1 ──────────────────▶ ┌──────────┐
└──────┘          │ Skip gate: F1*(1-U1)  │  Gate    │──▶ Decoder Block 4
┌──────┐          ▼                      └──────────┘
│Block2│──▶ F2 + U2 ──────────────────▶ ┌──────────┐
└──────┘          │                     │  Gate    │──▶ Decoder Block 3
┌──────┐          ▼                     └──────────┘
│Block3│──▶ F3 + U3 ──────────────────▶ ┌──────────┐
└──────┘          │                     │  Gate    │──▶ Decoder Block 2
┌──────┐          ▼                     └──────────┘
│Block4│──▶ F4 + U4 ──────────────────▶ ┌──────────┐
└──────┘                                │  Bridge  │──▶ Decoder Bottleneck
                                        └──────────┘
```

**Novelty claim:** No existing work propagates per-pixel uncertainty through ALL architectural levels. Deep Ensembles compute output-level variance. MC Dropout is approximate Bayesian inference at test time. FAUP-Net makes uncertainty a first-class architectural signal from encoder to output.

**Estimated effort:** 2–3 weeks (design + implementation + experiments + ablation)

---

### 3.2 Self-Supervised Pre-training (MAE / Contrastive)

**Concept:** Pre-train the encoder on unlabeled CT volumes using masked autoencoding or contrastive learning, then fine-tune for segmentation.

**Why:** Improves feature quality, especially in low-data regimes. Adds a "pre-training" contribution to the paper.

**Estimated effort:** 1–2 weeks

---

### 3.3 Test-Time Augmentation & 3D Sliding Window

**Concept:** 8× TTA (flips + rotations) at inference + 2D→3D stitch for per-volume metrics.

**Why:** Standard practice in SOTA papers. TTA typically gives +0.5–2% Dice. Per-volume metrics align with LiTS challenge evaluation.

**Estimated effort:** 2–3 days

---

## 4. Research Paper Titles by Direction

### Angle: Full-Architecture Uncertainty (FAUP-Net)

| # | Title | Venue Fit |
|---|-------|-----------|
| 1 | FAUP-Net: Full-Architecture Uncertainty Propagation for Reliable Medical Image Segmentation | MICCAI, IEEE TMI |
| 2 | Uncertainty as a First-Class Signal: Propagating Epistemic Uncertainty Through Every Level of a Segmentation Network | MICCAI, MedIA |
| 3 | Learning Where to Trust: Uncertainty-Gated Skip Connections in U-Net Ensembles | MICCAI, MIDL |
| 4 | Rethinking Uncertainty in Medical Image Segmentation: A Full-Architecture Propagation Framework | IEEE TMI, MedIA |

### Angle: Clinical / Application

| # | Title | Venue Fit |
|---|-------|-----------|
| 5 | Uncertainty-Guided Human-AI Triage for Liver Tumor Segmentation: A Simulation Study | Radiology AI, Lancet Digital Health |
| 6 | Calibrated Ensembles for Trustworthy Liver Tumor Segmentation: A Reliability Analysis | IEEE TMI, Medical Physics |
| 7 | LLM-Augmented Clinical Reporting from Automated Liver Tumor Segmentation | JAMIA, npj Digital Medicine |

### Angle: Efficiency / Edge

| # | Title | Venue Fit |
|---|-------|-----------|
| 8 | Knowledge Distillation of Uncertainty-Aware Ensembles for Edge-Deployable Liver Tumor Segmentation | IEEE JBHI, MICCAI Workshop |
| 9 | Data Efficiency in Medical Image Segmentation: How Much Data Does Your Model Really Need? | MIDL, Scientific Reports |

### Angle: Loss Function Novelty

| # | Title | Venue Fit |
|---|-------|-----------|
| 10 | UWACL++: Multi-Scale Uncertainty-Weighted Adaptive Compound Loss for Imbalanced Medical Image Segmentation | MICCAI, Pattern Recognition |

---

## 5. Movable / Swappable Node Map

Every component below can be swapped independently to form ablation experiments or alternative configurations:

```
┌───────────────────────┬──────────────┬────────────────────┬────────────────────┐
│      COMPONENT        │   DEFAULT    │     OPTION A       │     OPTION B       │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Self-supervised pre-  │ None         │ MAE (masked auto-  │ Contrastive (Sim-  │
│   training            │              │ encoder)           │ CLR style)         │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Encoder backbone      │ MobileNetV2  │ ResNet-34          │ EfficientNet-B0    │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Uncertainty estimator │ 2× light     │ MC Dropout heads   │ Block-level        │
│   (per block)         │ dropout heads│ (5 passes)         │ ensembles (heavy)  │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Skip gate type        │ Soft: (1-σ²) │ Hard: binary mask  │ Attention: learned │
│                       │              │ at threshold 0.5   │ attn × (1-σ²)      │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Decoder style         │ FAUP decoder │ Standard U-Net     │ Attention U-Net    │
│                       │              │ (ablation)         │ decoder            │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Ensemble strategy     │ Bagging      │ Backbone diversity │ Snapshot ensemble  │
│                       │ (same arch)  │ (MNV2+ResNet+Eff) │ (one training)     │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Loss function         │ UWACL-v2     │ UWACL-v1           │ Focal Tversky      │
│                       │ (multi-scale)│ (output only)      │                    │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Inference strategy    │ Mean ensemble│ Uncertainty-       │ TTA (8× aug) +     │
│                       │              │ weighted ensemble  │ mean ensemble      │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Post-processing       │ None         │ CRF refinement     │ Adaptive threshold │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Evaluation mode       │ Slice-wise   │ 3D sliding window  │ Per-volume only    │
│                       │              │ (stitch 2D→3D)     │                    │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Cross-validation      │ Single split │ 5-fold             │ 10-fold            │
├───────────────────────┼──────────────┼────────────────────┼────────────────────┤
│ Experiment tracking   │ MLflow       │ Weights & Biases   │ CSV logs only      │
└───────────────────────┴──────────────┴────────────────────┴────────────────────┘
```

---

## 6. Full Ablation Study Design

Required for any publication to quantify contribution of each component:

### 6.1 Model Architecture Ablation

| Experiment | Encoder | Gate | Decoder | Ensemble | Loss | Expected Dice |
|-----------|---------|------|---------|----------|------|--------------|
| 1. Single model | MobileNetV2 | None | Standard | None | Combined | Baseline |
| 2. + FAUP encoder | FAUP-MNV2 | None | Standard | None | Combined | +? |
| 3. + Gate | FAUP-MNV2 | Soft | Standard | None | Combined | +? |
| 4. + FAUP decoder | FAUP-MNV2 | Soft | FAUP | None | Combined | +? |
| 5. + Bagging (3×) | FAUP-MNV2 | Soft | FAUP | 3× bagging | Combined | +? |
| 6. + UWACL-v2 (full) | FAUP-MNV2 | Soft | FAUP | 3+1× | UWACL-v2 | Target |

### 6.2 Loss Function Ablation

| Loss Variant | Dice | IoU | ECE |
|-------------|------|-----|-----|
| Dice-only | ? | ? | ? |
| BCE-only | ? | ? | ? |
| Combined (Dice+BCE) | ? | ? | ? |
| Focal Tversky | ? | ? | ? |
| UWACL-v1 (output only) | ? | ? | ? |
| UWACL-v2 (multi-scale) | ? | ? | ? |

### 6.3 Gate Type Ablation

| Gate | Dice | HD95 | ECE |
|------|------|------|-----|
| No gate (baseline) | ? | ? | ? |
| Hard gate (binary) | ? | ? | ? |
| Soft gate (1-σ²) | ? | ? | ? |
| Attention + uncertainty | ? | ? | ? |
| Learnable α gate | ? | ? | ? |

### 6.4 Hyperparameter Sensitivity

| Parameter | Values to Test |
|-----------|---------------|
| β (UWACL weight scaling) | 1.0, 3.0, 5.0, 10.0 |
| Initial τ (temperature) | 0.01, 0.05, 0.1, 0.5 |
| τ schedule | linear_decay, cosine, exponential, constant |
| Ensemble size | 1, 2, 3, 5 |
| pos_weight | 5, 10, 20, 50 |
| Loss weights (dice:bce) | 0.3:0.7, 0.5:0.5, 0.7:0.3 |

---

## 7. Target Venues by Ambition

### Tier 1 (Flagship)

| Venue | Type | Pages | Match |
|-------|------|-------|-------|
| **MICCAI** | Conference (top) | 8 | FAUP-Net full architecture |
| **IEEE TMI** | Journal (top) | 12+ | FAUP + ablation + multi-dataset |
| **Medical Image Analysis (MedIA)** | Journal (top) | 12+ | FAUP + clinical validation |

### Tier 2 (Strong)

| Venue | Type | Pages | Match |
|-------|------|-------|-------|
| **MIDL** | Conference | 8 | Method-focused papers |
| **IEEE JBHI** | Journal | 10+ | Clinical + engineering |
| **Pattern Recognition** | Journal | 10+ | Method + loss novelty |
| **Scientific Reports** | Journal | ~10 | Smaller contributions |

### Tier 3 (Clinical / Digital Health)

| Venue | Type | Match |
|-------|------|-------|
| **Lancet Digital Health** | Journal | Human-AI triage study |
| **Radiology AI** | Journal | Clinical workflow |
| **npj Digital Medicine** | Journal | LLM reporting |
| **JAMIA** | Journal | Clinical informatics |

---

## Quick-Start Recommendation

**If you want something fast, novel, and easy to write up:**

| Priority | Project | Effort | Novelty | Venue Fit |
|----------|---------|--------|---------|-----------|
| 1 | Calibration study + reliability diagrams | 1–2 days | Medium | MICCAI, TMI |
| 2 | Human-AI triage simulation | 1–2 days | High | Lancet Digital Health |
| 3 | LLM-augmented clinical reporting | 1–2 days | High | npj Digital Medicine |
| 4 | Knowledge distillation | 2–3 days | Medium | IEEE JBHI |
| 5 | Low-data regime study | 2–3 days | Medium | MIDL, Sci Reports |

**Long-term:** FAUP-Net full architecture (2–3 weeks) → strongest novelty, flagship venue potential.

---

*Maintained as part of the Liver Tumor Segmentation & Analysis Platform.*
*Last updated: July 2026*

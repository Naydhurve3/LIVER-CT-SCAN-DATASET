# Liver Tumor Segmentation - Master Execution Plan

## Project Structure (Clean State)

```
Liver/
├── 📄 MASTER_PLAN.md                     ← You are here - Complete guide
├── 📄 README.md                          ← Project overview
├── 📄 COMPLETE_WORKFLOW.md               ← Detailed workflow
├── 📄 SHORT_WORKFLOW.md                  ← Quick flow
├── 📄 PROJECT_WORKFLOW.md               ← Visual workflow
├── 📄 PYTORCH_EXECUTION_GUIDE.md        ← PyTorch guide
├── 📄 PROJECT_UPGRADE_ANALYSIS.md       ← Old vs New comparison
├── 📄 EXECUTION_GUIDE.md                 ← Step-by-step guide
└── 📁 docs/
    └── FULL_ARCHITECTURE.md              ← Architecture doc
```

---

# RESEARCH NOTEBOOK STRUCTURE

This project follows a **research-oriented notebook architecture** for better reproducibility and clarity.

---

## PART 1: CORE ANALYSIS PIPELINE (Required)

### Notebook 01: eda_analysis
```
PURPOSE: Exploratory Data Analysis

WHAT IT DOES:
├── Load dataset and compute statistics
│   ├── Volume dimensions
│   ├── Voxel spacing
│   └── Slice counts per volume
├── Generate intensity histograms for each CT scan
├── Visualize liver/tumor volume distributions
├── Plot 3D liver shape variability across patients
├── Create class imbalance charts
│   ├── Background vs liver vs tumor voxels
│   └── Tumor slice ratio (currently ~12%)
├── Identify outliers or problematic cases
└── Generate data/metadata/statistics.json

OUTPUTS: statistics.json, distribution plots, outlier list
```

### Notebook 02: preprocessing_pipeline
```
PURPOSE: Data Preprocessing and Enhancement

WHAT IT DOES:
├── Implement HU windowing with configurable ranges
│   ├── Window: [-100, 400] HU (liver tissue)
│   └── Alternative: [-200, 500] HU
├── Create resampling function for isotropic spacing
├── Build normalization routines
│   ├── Min-Max normalization (0-1)
│   └── Z-score normalization
├── Experiment with 2D contrast enhancement
│   ├── CLAHE (Contrast Limited Adaptive Histogram Equalization)
│   └── Compare before/after preprocessing
├── Visualize before/after for multiple patients
└── Save preprocessed data in efficient format

OUTPUTS: Preprocessed images ready for training
```

### Notebook 03: feature_extraction
```
PURPOSE: Extract Traditional and Deep Features

WHAT IT DOES:
├── Extract first-order statistics
│   ├── Mean, std, skewness, kurtosis
│   ├── From liver and tumor regions separately
├── Compute GLCM texture features
│   ├── Contrast, energy, homogeneity, correlation
├── Calculate shape descriptors
│   ├── Volume, surface area, sphericity
├── Extract deep learning features
│   ├── Using pretrained models (ResNet, VGG)
├── Create feature matrices for downstream analysis
└── Save feature matrices for comparison

OUTPUTS: Feature matrices (.npy or .csv)
```

---

## PART 2: MODEL DEVELOPMENT

### Notebook 04: unet_2d_experiments
```
PURPOSE: Train 2D U-Net Models

WHAT IT DOES:
├── Load preprocessed data (from Notebook 02)
├── Define 2D U-Net architecture
│   ├── Encoder: MobileNetV2 / EfficientNet
│   ├── Decoder: UNet-style with skip connections
│   └── Output: (256, 256, 1) sigmoid
├── Implement data generators for slice-based training
├── Set up training loop
│   ├── Loss: BCE + Dice combined
│   ├── Optimizer: Adam
│   ├── Two-phase: frozen encoder → fine-tune all
├── Track metrics during training
│   ├── Dice score, IoU, loss curves
├── Visualize sample predictions on validation cases
└── Save best model checkpoints

MODELS TO TRY:
├── MobileNetV2 + UNet (3.4M params, 0.6GB VRAM) ← RECOMMENDED
├── EfficientNetB0 + UNet (5.3M params, 0.9GB VRAM)
├── Attention UNet (9M params, 1.0GB VRAM)
└── UNet++ (6M params, 1.2GB VRAM)

OUTPUTS: Trained models, training curves, sample predictions
```

### Notebook 05: unet_3d_experiments
```
PURPOSE: Train 3D U-Net Models (Optional - More Compute)

WHAT IT DOES:
├── Implement 3D patch-based sampling (due to memory constraints)
│   ├── Patch size: 128×128×64
│   └── Stride: 64×64×32 for overlap
├── Build 3D U-Net architecture
│   ├── 3D convolutions
│   ├── 3D pooling
│   └── 3D upsample
├── Configure 3D data augmentation
│   ├── 3D rotations
│   ├── 3D elastic deformations
├── Train and monitor GPU memory usage
├── Compare convergence behavior against 2D version

NOTE: Requires more VRAM (~8GB+), skip if limited resources

OUTPUTS: 3D trained models
```

### Notebook 06: nnuet_setup_and_training
```
PURPOSE: Use nnU-Net Framework (Optional)

WHAT IT DOES:
├── Install and configure nnU-Net framework
├── Preprocess dataset into nnU-Net format
├── Run experiment planning and preprocessing
│   ├── Auto-detection of optimal configuration
│   └── Dataset fingerprint generation
├── Execute 5-fold cross-validation training
├── Extract and save results from all folds

NOTE: State-of-the-art self-configuring segmentation framework

OUTPUTS: nnU-Net results, leaderboard-style comparison
```

---

## PART 3: EVALUATION AND ANALYSIS

### Notebook 07: model_evaluation_metrics
```
PURPOSE: Comprehensive Model Evaluation

WHAT IT DOES:
├── Calculate Dice scores for all models
├── Compute Hausdorff Distance (95th percentile)
├── Generate precision and recall curves
├── Create confusion matrices per patient
├── Build comprehensive results table
│   ├── Compare: MobileNetV2, EfficientNet, Attention UNet, etc.
│   └── Columns: Dice, IoU, HD95, Precision, Recall
└── Statistical significance testing

METRICS COMPUTED:
├── Dice Coefficient (primary)
├── IoU / Jaccard Index
├── Sensitivity (Tumor Detection Rate)
├── Specificity (Background Accuracy)
└── Hausdorff Distance (95th percentile)

OUTPUTS: metrics.csv, comparison tables
```

### Notebook 08: benchmark_comparison
```
PURPOSE: Compare Against Published Baselines

WHAT IT DOES:
├── Reproduce LiTS baseline preprocessing from original paper
├── Train baseline 3D U-Net as reference
├── Load and compare against published leaderboard results
├── Create comparative visualization
│   ├── Bar charts for all metrics
│   └── Box plots across cross-validation folds
├── Calculate statistical significance between methods
└── Generate final comparison figure for paper

BASELINES TO COMPARE:
├── LiTS Challenge Leaderboard (top methods)
├── Traditional U-Net (from scratch)
└── Your pretrained models

OUTPUTS: Benchmark comparison figures, statistical tests
```

### Notebook 09: error_analysis
```
PURPOSE: Deep Dive into Model Failures

WHAT IT DOES:
├── Identify worst-performing cases for each model
├── Analyze performance stratified by tumor size
│   ├── Small tumors (< 1000 pixels)
│   ├── Medium tumors (1000-10000 pixels)
│   └── Large tumors (> 10000 pixels)
├── Visualize failure modes
│   ├── Over-segmentation cases
│   ├── Under-segmentation cases
│   ├── Missing tumors entirely
├── Correlate errors with patient-specific characteristics
├── Generate error heatmaps across liver anatomy
└── Document findings for improvement

OUTPUTS: Error analysis report, failure case visualizations
```

---

## PART 4: ABLATION STUDIES

### Notebook 10: ablation_window_levels
```
PURPOSE: Study HU Window Impact

WHAT IT DOES:
├── Train identical model with different HU window ranges
├── Compare performance across windows
│   ├── [-100, 400] HU (liver tissue) ← DEFAULT
│   ├── [-200, 500] HU (wider)
│   └── [-50, 200] HU (narrow)
├── Analyze which structures are enhanced by each window
├── Create composite multi-channel inputs if beneficial
└── Select optimal window for liver/tumor segmentation

OUTPUTS: Window comparison metrics, recommendations
```

### Notebook 11: ablation_augmentation
```
PURPOSE: Study Data Augmentation Impact

WHAT IT DOES:
├── Train with and without each augmentation type
├── Test augmentation types individually
│   ├── Horizontal flip
│   ├── Vertical flip
│   ├── Rotation (90°)
│   ├── Elastic deformations (liver shape variation)
│   └── Brightness/contrast adjustment
├── Measure generalization gap
│   ├── Train vs validation performance
│   └── With vs without augmentation
└── Design optimal augmentation strategy

OUTPUTS: Augmentation impact analysis, best strategy
```

### Notebook 12: ablation_loss_functions
```
PURPOSE: Compare Loss Functions

WHAT IT DOES:
├── Compare different loss functions
│   ├── Binary Cross Entropy (BCE)
│   ├── Dice Loss
│   ├── BCE + Dice (combined) ← DEFAULT
│   ├── Tversky Loss (α=0.3, β=0.7)
│   └── Focal Tversky Loss
├── Experiment with Tversky parameters
│   ├── α (FP weight): 0.3, 0.5, 0.7
│   ├── β (FN weight): 0.5, 0.7, 0.9
├── Generate loss landscapes or convergence plots
├── Select optimal loss based on precision-recall trade-off
└── Analyze impact on tumor detection vs false positives

OUTPUTS: Loss function comparison, recommendations
```

---

## PART 5: RESEARCH SYNTHESIS

### Notebook 13: results_visualization_dashboard
```
PURPOSE: Create Publication-Quality Figures

WHAT IT DOES:
├── Create comprehensive figure collection
│   ├── Qualitative results (side-by-side)
│   ├── Quantitative results (bar charts)
│   ├── Training curves
│   └── Error analysis plots
├── Generate side-by-side predictions
│   ├── Ground truth vs model output
│   ├── Multiple patients, multiple models
├── Build 3D renderings of segmentations
├── Produce metric distribution plots
│   └── Box plots across cross-validation folds
└── Assemble final comparison tables

OUTPUTS: All figures for paper/presentation
```

### Notebook 14: reproducibility_report
```
PURPOSE: Document Everything for Reproducibility

WHAT IT DOES:
├── Document all hyperparameters and random seeds
├── Create environment specification
│   ├── Python version
│   ├── Package versions (requirements.txt)
│   └── GPU specifications
├── Write markdown explanations
│   ├── Each design decision
│   ├── Rationale for choices
├── Generate final performance summary
├── List limitations and potential improvements
└── Prepare for open-source release

OUTPUTS: README, requirements.txt, documentation
```

---

## PART 6: OPTIONAL ADVANCED STUDIES

### Notebook 15: ensemble_methods (Optional)
```
WHAT IT DOES:
├── Implement simple averaging of 2D and 3D model predictions
├── Test weighted voting based on tumor size
├── Create rule-based ensemble for edge cases
└── Evaluate ensemble performance gain

EXPECTED: +2-5% Dice improvement over single models
```

### Notebook 16: clinical_feature_correlation (Optional)
```
WHAT IT DOES:
├── Correlate extracted texture features with segmentation difficulty
├── Analyze if patient metadata predicts performance
├── Build simple classifier using traditional features
└── Compare to deep learning approach
```

### Notebook 17: domain_adaptation_test (Optional)
```
WHAT IT DOES:
├── Test trained models on external dataset (if available)
├── Analyze performance drop on new hospital data
├── Implement simple adaptation techniques
└── Document generalization capabilities
```

---

# EXECUTION ORDER

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         RECOMMENDED WORKFLOW ORDER                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  PHASE 1: FOUNDATIONAL (Start here)                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  Notebook 01: eda_analysis                                                   │     │
│  │       ↓                                                                      │     │
│  │  Notebook 02: preprocessing_pipeline                                         │     │
│  │       ↓                                                                      │     │
│  │  Notebook 03: feature_extraction (optional)                                 │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│  PHASE 2: MODEL TRAINING                                                           │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  Notebook 04: unet_2d_experiments      ← MAIN MODEL                        │     │
│  │       ↓                                                                      │     │
│  │  Notebook 05: unet_3d_experiments      ← Optional (more compute)           │     │
│  │       ↓                                                                      │     │
│  │  Notebook 06: nnuet_setup             ← Optional                            │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│  PHASE 3: EVALUATION                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  Notebook 07: model_evaluation_metrics                                     │     │
│  │       ↓                                                                      │     │
│  │  Notebook 08: benchmark_comparison                                         │     │
│  │       ↓                                                                      │     │
│  │  Notebook 09: error_analysis                                                │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│  PHASE 4: ABLATION STUDIES                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  Notebook 10: ablation_window_levels                                        │     │
│  │  Notebook 11: ablation_augmentation                                         │     │
│  │  Notebook 12: ablation_loss_functions                                       │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│  PHASE 5: SYNTHESIS                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐     │
│  │  Notebook 13: results_visualization_dashboard                              │     │
│  │  Notebook 14: reproducibility_report                                       │     │
│  └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│  OPTIONAL:                                                                         │
│  ├── Notebook 15: ensemble_methods                                                 │
│  ├── Notebook 16: clinical_feature_correlation                                   │
│  └── Notebook 17: domain_adaptation_test                                         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

# KEY DECISIONS TO MAKE

| Decision | Options | Recommendation |
|----------|---------|----------------|
| **Start with** | 2D or 3D U-Net | Start with 2D (less compute) |
| **Primary Model** | Which backbone | MobileNetV2 (3.4M, 0.6GB) |
| **Loss Function** | BCE, Dice, Combined | BCE + Dice |
| **Augmentation** | Basic or Advanced | Start basic, ablate later |
| **Preprocessing** | With or without CLAHE | Use CLAHE |
| **Evaluation** | Single or Ensemble | Start single, add ensemble later |

---

# WHAT TO CREATE NEXT

Based on the notebook structure, the **first Python file to create**:

```
Notebook 01: eda_analysis
└─→ Need: src/01_eda_analysis.py or notebooks/01_eda_analysis.ipynb
```

This will:
1. Load the dataset
2. Compute statistics
3. Generate statistics.json
4. Create visualization plots

---

# CURRENT STATE

```
Liver/
├── 📄 8 MD files                    ← Documentation ✅
└── (All code files removed - will recreate step by step)
```

**Ready to create first notebook!**

Would you like me to create **Notebook 01: eda_analysis** now?
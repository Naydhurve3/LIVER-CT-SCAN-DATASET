# Interpreting Segmentation Results — An Analyst's Guide

## Key Metrics

### Dice Similarity Coefficient
- **Range**: 0 (no overlap) to 1 (perfect overlap)
- **Clinical threshold**: >0.8 is generally considered good for liver segmentation
- **Interpretation**: Measures spatial overlap between predicted and ground-truth masks

### IoU (Intersection over Union) / Jaccard Index
- **Range**: 0 to 1
- **Relationship to Dice**: IoU = Dice / (2 - Dice)
- **Use case**: More stringent than Dice for small objects

### Per-Slice Metrics
- Important to evaluate per-slice performance because tumors may be present in only a few slices
- A high overall Dice can mask poor performance on tumor-positive slices

## Model Comparison Framework

| Metric | What It Tells You |
|--------|-------------------|
| Dice (overall) | General segmentation quality |
| Dice (tumor slices only) | Performance on clinically relevant slices |
| Precision | How many predicted tumor pixels are correct |
| Recall | How many true tumor pixels are detected |
| F1 Score | Harmonic mean of precision and recall |
| Hausdorff Distance | Boundary accuracy |

## Common Pitfalls

1. **Split leakage** — Slices from the same volume in different splits (yours is volume-wise, so this is handled correctly)
2. **Class imbalance** — With 937:1 imbalance, a model predicting all-background gets 99.9% accuracy but 0 Dice
3. **Volume-level bias** — Your val/test volumes are systematically larger; consider stratified splitting
4. **Overfitting to liver shape** — Model may learn liver boundary well but miss small tumors within

# Spatial Orientation Forensics, Denominator Reconciliation & Quality Audit

[![Audit](https://img.shields.io/badge/Audit-Passed-success.svg)]()
[![Containment](https://img.shields.io/badge/ROI%20Containment-100%25-brightgreen.svg)]()

## Executive Overview

Deep learning models trained on medical imaging are uniquely sensitive to spatial alignment errors. During dataset auditing, multi-stage forensics uncovered **47 volumes with $180^\circ$ in-plane rotation discrepancies** and a **$4\times$ tumor burden denominator mismatch** in legacy preprocessing scripts.

---

## 1. Spatial Orientation Flips & Matrix Corrections

Auditing revealed that CT slice images and target segmentation masks in Kaggle Part 2 and Hugging Face mirrors suffered from a $180^\circ$ in-plane rotation flip.

```
       RAW ANOMALOUS ALIGNMENT (47 Volumes)           CORRECTED SPATIAL ALIGNMENT
       ┌───────────────────────────────────┐          ┌───────────────────────────────────┐
       │   Image Matrix    Mask Matrix     │          │   Image Matrix    Mask Matrix     │
       │     (Normal)       (Flipped)      │ ───────► │     (Normal)       (Normal)       │
       │     [  ▲  ]         [  ▼  ]       │ Rot180°  │     [  ▲  ]         [  ▲  ]       │
       └───────────────────────────────────┘          └───────────────────────────────────┘
```

### Applied Spatial Transform Mapping:
- **`Identity` Matrix (84 Volumes)**: Volumes `0–82` and `100` (Original orientation retained).
- **`Rot180` Matrix (47 Volumes)**: Volumes `83–99` and `101–130` ($180^\circ$ planar rotation applied).

```
Volume Alignment Visual Audit:
```
![Volume Progression](../figures/volume_progression.png)

---

## 2. 4x Tumor Burden Denominator Reconciliation

Legacy dataset scripts calculated the volume tumor burden percentage using a $512\times 512$ pixel denominator ($262,144\text{ px}$) against $256\times 256$ native masks ($65,536\text{ px}$):

$$\text{Legacy Burden \%} = \frac{\text{Tumor Pixels (256x256)}}{512 \times 512 = 262,144} \times 100 \quad (\text{Deflated } 4\times)$$

$$\text{Reconciled Burden \%} = \frac{\text{Tumor Pixels (256x256)}}{256 \times 256 = 65,536} \times 100 \quad (\text{True Native Burden})$$

* **Impact**: True mean training tumor burden was reconciled from `0.0251%` to **`0.1003%`**.

---

## 3. 2-Stage Bounding Box & ROI Containment Verification

To crop out unneeded background abdominal tissue, a 2-stage ROI extraction protocol was deployed:
1. Predict Stage-1 Liver probability mask at $0.50$ threshold.
2. Select the `largest_3d` connected liver component.
3. Apply $+16\text{px}$ spatial padding around the bounding box $[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$.
4. Crop and resize to $256\times 256$.

![Tumor Bounding Box Analysis](../figures/tumor_bbox_analysis.png)

* **Containment Verification**: Achieved **100% containment** of all tumor pixels across Train and Validation (`152,763 / 152,763` pixels in volume `V116`).

---

[Previous: Provenance](DATA_PROVENANCE_AND_ACQUISITION.md) | [Back to Main README](../README.md) | [Next: Radiometrics & HU Windowing](RADIOMETRICS_AND_HU_WINDOWING.md)

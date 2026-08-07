# External Evaluation Protocol: 3D IRCADb-01 Benchmark

[![External Val](https://img.shields.io/badge/External%20Val-3D%20IRCADb--01-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Frozen%20Contract-green.svg)]()

## 1. Protocol Overview

To evaluate out-of-domain cross-center generalization without risking data leakage on the sealed LiTS test set, the **3D IRCADb-01** public benchmark dataset was integrated (`step_12` through `step_17` in `mark 1 (part 2)`).

```
        Raw DICOM (IRCAD) ──► NIfTI Conversion ──► HU [-160, +240] ──► 256x256 Parity Build
```

- **Origin**: IRCAD (Research Institute against Digestive Cancer), France.
- **Volume Count**: **20 anonymized abdominal CT scans** ($2,827$ axial slices).
- **Lesion Annotations**: 3D manual segmentations of liver parenchyma, hepatic tumors, portal veins, and adjacent organs.

---

## 2. Parity & Standardization Pipeline

To maintain parity with the LiTS training pipeline, IRCAD scans are converted through identical preprocessing:
1. Resampled in-plane to $256\times 256$.
2. Standardized using broad abdominal window `[-160, +240] HU`.
3. Evaluated un-tuned under frozen Mark 4E Checkpoint Fusion policy.

---

[Previous: Model Benchmarks](MODEL_BENCHMARKS_AND_FUSION_POLICY.md) | [Back to Main README](../README.md)

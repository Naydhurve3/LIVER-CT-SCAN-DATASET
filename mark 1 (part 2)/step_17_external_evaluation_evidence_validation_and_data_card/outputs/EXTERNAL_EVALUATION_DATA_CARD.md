# 3D-IRCADb-01 External Evaluation Data Card

## Result

- Assessment: `EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS`
- Frozen Step 16 gate: **PASS** (11/11 mandatory rows)
- Cohort: 20 patients; 15 tumour-positive; 5 tumour-negative; zero exclusions
- Run UUID: `61d1f140-c8b4-436a-928a-1e4b6f7c0b56`

## Primary evidence

- Global Dice: 0.847977
- Mean positive-patient Dice: 0.711196 (95% patient-bootstrap CI 0.567803–0.827823)
- Median positive-patient Dice: 0.835514
- Minimum positive-patient Dice: 0.012992
- Q1 positive-slice detection: 50.00%
- Positive predicted-empty rate: 9.86%
- Empty-slice false-positive rate: 11.57%

## Required caveats

- Worst positive case: `ircadb_18` with Dice 0.012992.
- All 5 tumour-negative patients had at least one predicted pixel; total standardized physical FP volume was 52.936 ml.
- Smallest train-derived lesion stratum detection was 63.64% (14/22 components).
- Tumour-negative labels exclude adrenal/generic non-hepatic tumour folders; prediction biology cannot be inferred from this evaluation.
- The failure atlas shows visible low-attenuation structures in the largest negative-control prediction cases, but only expert/source-annotation review can determine their biological meaning; they remain false positives under the frozen accepted truth.
- The reported all-pixel reliability curve and ECE are background-dominated descriptive diagnostics, not a clinical calibration claim.
- The result does not establish clinical readiness or universal generalization.

## Integrity

- Step 16 signature and all 32 evidence/cache inventory rows independently verified.
- Headline metrics independently recomputed from saved counts.
- No inference, tuning, threshold sweep, case exclusion or Step 16 rerun occurred in Step 17.

# Step 17 — External-evaluation evidence validation and data card

Open `step_17.ipynb` and use **Restart Kernel and Run All** from this directory.

This phase is read-only with respect to the sealed Step 16 result. It verifies signatures and cache hashes, independently recomputes metrics, analyzes negative controls and difficulty strata, creates probability diagnostics and a failure atlas, and writes a signed external-evaluation data card under `outputs/`.

It performs no model loading, inference, tuning, threshold sweep, case exclusion, Step 16 rerun or local LiTS test access.

## Verified execution — 5 August 2026

- Result level: `EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS`.
- Independently verified all 18 Step 16 signed artifacts and all 32 evidence/cache inventory rows; recomputed every headline metric and all 11 acceptance decisions exactly.
- Confirmed Step 16's frozen external gate pass: global Dice 0.847977; mean positive-patient Dice 0.711196 (95% CI 0.567803–0.827823); median 0.835514; minimum 0.012992.
- Quantified caveats: `ircadb_18` Dice 0.012992; smallest-lesion detection 14/22 (63.64%); all five negative controls had predictions totaling 52.936 ml on the standardized physical grid.
- The largest negative-control predictions occurred in cases 7 and 14. Failure-atlas review shows visible low-attenuation structures, but biological meaning remains unresolved and the predictions stay false positives under the frozen accepted truth.
- The all-pixel reliability plot is explicitly labeled background-dominated and is not treated as a clinical calibration claim.
- Produced a signed data card, validated-findings table, patient-risk profile, physical-volume negative-control audit, lesion analysis, probability diagnostics, dashboard and failure atlas.
- Step 17 signature: `4b0f2b13eef3f5a70a41eca94b826703f3daeac85b17b354886c3357ac4c3b48`.
- Inference performed: false. Step 16 rerun: false. Tuning performed: false. Local LiTS test accessed: false.
- Next action: optional read-only expert/source-annotation review of the preserved failure atlas, or contract a second independent dataset. Do not tune or rerun Step 16.

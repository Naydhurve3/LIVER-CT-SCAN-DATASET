# Liver Tumor Segmentation Project Record

This folder is the human-readable record of the work performed in `Practice/` through 29 July 2026. It records the dataset, validation rules, experiment sequence, parameters, results, failures, artifacts, and current decision.

## Current status

- The corrected manifest-driven dataset and 16-slice overfit gates passed.
- The five-epoch baseline smoke test passed.
- Every longer validation experiment has failed at least one patient or detection guardrail.
- The strongest mean positive-patient Dice observed so far was approximately `0.4069` from the organ-assisted intensity-normalized experiment.
- That experiment is diagnostic rather than deployable because ground-truth organ masks were used to calculate normalization statistics.
- The deployable image-only multi-task experiment completed 10 epochs but failed: best mean patient Dice `0.3329`, volume 104 and 116 Dice approximately zero, Q1 detection `28.14%`.
- The test split has remained locked and no saved gate reports test-image access.
- The immediate next step is validation-only probability calibration of the epoch-8 multi-task checkpoint. Do not start another long training run or open the test split first.

## Documents

1. [01_dataset_and_provenance.md](01_dataset_and_provenance.md) — authoritative data build, schema, dimensions, splits, masks, orientation, and hashes.
2. [02_validation_and_metric_contract.md](02_validation_and_metric_contract.md) — leakage controls, metrics, lesion groups, thresholds, and pass/fail rules.
3. [03_experiment_timeline.md](03_experiment_timeline.md) — chronological record of every major notebook and decision.
4. [04_parameters_and_techniques.md](04_parameters_and_techniques.md) — models, losses, preprocessing, samplers, training, thermal controls, and post-processing.
5. [05_complete_results.md](05_complete_results.md) — consolidated numeric results from all saved gate files.
6. [06_failures_fixes_and_lessons.md](06_failures_fixes_and_lessons.md) — runtime errors, scientific failures, fixes, and what not to repeat.
7. [07_artifact_and_notebook_map.md](07_artifact_and_notebook_map.md) — notebooks, generators, checkpoints, CSV files, JSON decisions, and figures.
8. [08_current_plan.md](08_current_plan.md) — current interpretation and ordered next-step plan.
9. [CHAT_HANDOFF_SUMMARY.md](CHAT_HANDOFF_SUMMARY.md) — portable full-context summary for another chat or research advisor.
10. [CURRENT_PROBLEMS_AND_BLOCKERS.md](CURRENT_PROBLEMS_AND_BLOCKERS.md) — problem-focused briefing with evidence, ruled-out approaches, runtime errors, and decisions needed.

## Source-of-truth hierarchy

1. Corrected manifest and source hashes.
2. `Practice/*_outputs/*gate_result.json` and status JSON files.
3. Saved CSV histories and patient/slice metrics.
4. Executed notebooks and figures.
5. These Markdown summaries.

If a number in these notes conflicts with a saved gate JSON or CSV, the saved machine-readable artifact wins and this documentation should be corrected.

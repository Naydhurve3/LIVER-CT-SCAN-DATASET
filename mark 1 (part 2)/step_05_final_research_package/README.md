# Step 05 — Final Research Package

## Purpose

This no-inference phase verifies the sealed Step 01–04 evidence and produces the final technical report, internal paper draft, figures, methods table, experiment timeline, decision log, failure analysis, reproducibility instructions, data card, checksum inventory and final project gate.

`FINAL_PROJECT_COMPLETE` means the research package is complete. It does not mean the model passed formal acceptance: V121 failed the predeclared minimum-patient Dice floor.

## Verified completion — 5 August 2026

- Result level: `FINAL_PROJECT_COMPLETE`.
- Package requirements: 12/12 passed.
- Formal model acceptance: failed; 7/8 final test/integrity targets passed.
- Failure preserved: minimum positive-patient Dice was effectively zero for V121 versus the frozen 0.01 floor.
- Step 04 evidence: 34 source/package artifacts inventoried with no checksum failure.
- Notebook: `nbformat` valid, all eight code cells parse, full no-inference execution completed with zero cell errors.
- Outputs: 23/23 required package artifacts exist.
- Test source files reopened in Step 05: false.
- Test inference rerun: false.
- Final disposition: archive and report; no further test use.

## Execution boundary

The notebook reads only sealed Part 2 outputs and signed Step 04 caches. It does not reopen dataset source images or masks, instantiate a loader or model, run inference, change the frozen policy, or rerun the test evaluation.

## Run

Use **Restart Kernel and Run All** with the project Python environment. The notebook is lightweight and CPU-only. All generated artifacts are written to `outputs/`.

## Final disposition

Archive and report the failed formal model acceptance. Manuscript revision may continue from the sealed outputs, but the Step 04 test run must never be repeated or used for model selection.

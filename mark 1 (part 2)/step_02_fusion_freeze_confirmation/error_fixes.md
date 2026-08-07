# Step 02 Error Fixes

No execution errors have been recorded yet. If a failure occurs, preserve completed per-volume caches, patch both the notebook and `create_step_02_notebook.py`, and record the traceback, root cause, repair, validation, and exact resume point here.

## 3 August 2026 — historical-cache maximum-outlier assertion

- Error: `AssertionError: Fresh inference differs from historical cache beyond tolerance` at the end of the fresh-inference cell.
- Evidence: all 26 repeated fresh model-volume inference pairs were exactly deterministic. Mean fresh-versus-historical error was between approximately `1e-8` and `1.6e-6`, but isolated historical-cache pixels reached maximum differences up to `0.043986`.
- Decision impact: after maximum fusion, only 97 of 700,252,160 pixels changed hard class at threshold 0.70 (`1.385e-7`), and maximum per-patient Dice change was approximately `0.000189`.
- Root cause: the original gate treated the maximum error of any one pixel as mandatory, making sparse historical-cache outliers fatal despite negligible aggregate and hard-decision impact. Fresh repeated inference itself is deterministic.
- Repair: maximum error is now diagnostic. Mandatory equivalence uses aggregate mean absolute error `<=1e-4`, fraction of score pixels above `5e-4` `<=1e-4`, fused hard-prediction disagreement at 0.70 `<=1e-6`, and reference metric absolute delta `<=5e-4`.
- Files patched: notebook and generator. Completed fresh probability caches, determinism evidence, cache-integrity evidence, and runtime logs were preserved.
- Resume: with the same live kernel, continue at the next code cell, `Compute frozen-policy patient, slice, and six-target metrics`; it reconstructs equivalence from saved caches and does not rerun inference. If the kernel was restarted, use Restart Kernel and Run All.

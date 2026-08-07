# Error fixes

## 2026-08-05 — manuscript figure visual QA

- Issue: the comparison legend overlapped global-Dice value labels, and the distribution figure displayed boxplot fliers on top of the complete jittered patient layer.
- Root cause: default in-axes legend placement and default `showfliers=True` were unsuitable for the final chart composition.
- Repair: moved the legend into unused upper-center space and disabled duplicate boxplot fliers while retaining every patient as a labelled-layer point.
- Files patched: both generated notebooks via `create_step_19_notebook.py`.
- Resume: restart and Run All. No model inference or source test access occurs.

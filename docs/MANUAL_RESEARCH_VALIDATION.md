# Manual Research Validation

The GPU runs are intentionally manual so temperature and interruption can be
managed on the RTX 3050 Ti laptop. Training is epoch-resumable; `--epochs` is
the total target, not the number of additional epochs.

## Resume the baseline to epoch 5

```powershell
.\.venv\Scripts\python.exe tools\train.py `
  --config configs\experiments\research_baseline.yaml `
  --resume models\research_validation\baseline_focal_dice\last_checkpoint.pth `
  --epochs 5
```

The existing epoch-1 checkpoint predates exact RNG checkpointing. Its model,
optimizer, history, best score, and cosine scheduler position are recovered;
the manifest labels its resume reproducibility as
`legacy_scheduler_reconstructed_rng_unavailable`. Every checkpoint written
after resuming contains exact scheduler, scaler, sampler, and RNG state.

Monitor temperature in a second PowerShell window:

```powershell
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used `
  --format=csv -l 5
```

If temperature remains above 87 C, press `Ctrl+C`. The trainer saves
`interrupted_checkpoint.pth` from the start of the interrupted epoch, so the
epoch can be repeated cleanly after cooling below 60 C:

```powershell
.\.venv\Scripts\python.exe tools\train.py `
  --config configs\experiments\research_baseline.yaml `
  --resume models\research_validation\baseline_focal_dice\interrupted_checkpoint.pth `
  --epochs 5
```

## Validation-only evaluation

```powershell
.\.venv\Scripts\python.exe tools\evaluate.py `
  --config configs\experiments\research_baseline.yaml `
  --checkpoint models\research_validation\baseline_focal_dice\best_checkpoint.pth `
  --split val `
  --output-dir experiments\research_validation\baseline_focal_dice\val_evaluation
```

Share `history.json`, `threshold_selection.json`, `evaluation.json`, and
`per_volume.csv`. Do not run the test split until all model and threshold
selection is locked.

## Candidate commands

Run these only after the validation promotion gate passes:

```powershell
.\.venv\Scripts\python.exe tools\train.py `
  --config configs\experiments\research_faup_focal.yaml `
  --epochs 5
```

```powershell
.\.venv\Scripts\python.exe tools\train.py `
  --config configs\experiments\research_faup_uwacl.yaml `
  --epochs 5
```

Dry runs automatically use unique timestamped model and result directories.
For any other independent run, pass a unique `--run-dir` and `--output-dir`;
the CLI refuses to overwrite an existing manifest unless `--resume` is used.

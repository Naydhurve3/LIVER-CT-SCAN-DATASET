# ADR-003: Full Determinism and Reproducibility

## Status
Accepted

## Context
Reproducibility is a core requirement for publication-grade research. Every experiment must produce identical results when run with the same config on the same hardware profile.

## Decision
- `reproducibility.py` centralizes seed setting across `random`, `numpy`, `torch`, and `cudnn`
- `MEDSEGX_DETERMINISTIC` env var (default `1`) enables deterministic algorithms
- Every experiment config includes a `seed` field (default `42`)
- MLflow (or W&B) tracks all hyperparameters, metrics, and artifact paths per run
- Held-out test set is touched exactly once per experiment

## Consequences
- Deterministic mode may reduce GPU throughput (~10–20%); non-deterministic mode available via env var
- Every experiment is replayable from its YAML config + seed alone

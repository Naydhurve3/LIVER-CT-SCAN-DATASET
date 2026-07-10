# ADR-001: Core Module with Zero External Dependencies

## Status
Accepted

## Context
MedSegX requires a foundational layer shared by all framework and research modules. To minimize coupling and ensure maintainability, this layer must have no dependencies on other MedSegX modules, PyTorch, or any third-party libraries beyond the Python standard library.

## Decision
Create `src/framework/core/` containing:

- `constants.py` — project-wide constants (paths, dataset splits, hyperparameters)
- `exceptions.py` — typed exception hierarchy (ConfigError, DatasetError, ModelError, etc.)
- `interfaces.py` — abstract base classes (Configurable, Trainable, Evaluable, Predictable, Plottable)
- `registry.py` — generic Registry class + named singletons (MODELS, LOSSES, METRICS, DATASETS, ...)
- `config.py` — YAML config loader with env-var resolution and inheritance merging
- `reproducibility.py` — seed setting + deterministic flag
- `factory.py` — builder functions that dispatch via registries

## Consequences
- Core is importable without any third-party libraries
- Registry pattern eliminates all if/elif chains for model/loss/metric selection
- All other modules depend on core; core depends on nothing

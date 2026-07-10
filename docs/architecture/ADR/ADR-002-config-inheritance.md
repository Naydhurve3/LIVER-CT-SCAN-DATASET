# ADR-002: Layered Config Inheritance

## Status
Accepted

## Context
Experiments require composing settings from datasets, models, training, augmentation, and evaluation. Flat configs are repetitive and error-prone. A layered inheritance model reduces duplication and ensures consistency.

## Decision
Configs inherit in a fixed order: `defaults/*` → `datasets/*` → `models/*` → `experiments/*`. CLI overrides apply last.

An `inherits` key in any YAML file lists parent configs resolved by `config.py:resolve_inherits()`. Environment variables are resolved with `${env:VAR_NAME}` syntax.

Example hierarchy for `baseline.yaml`:
```
baseline.yaml
├── defaults/training.yaml
├── defaults/optimizer.yaml
├── defaults/augmentation.yaml
├── defaults/preprocessing.yaml
├── defaults/evaluation.yaml
├── datasets/lits.yaml
└── models/mobilenetv2_unet.yaml
```

## Consequences
- Single source of truth for each config layer
- Experiments declare only their delta from defaults
- `build_experiment_config()` composes the full config at runtime

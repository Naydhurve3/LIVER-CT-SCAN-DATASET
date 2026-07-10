# ADR-007: Framework / Research Separation

## Status
Accepted

## Context
MedSegX serves dual purposes: (1) a stable framework for reproducible medical image segmentation experiments and (2) a sandbox for novel research contributions (UP³RE-Net, FAUP-Net, UWACL). These have different stability requirements — the framework must not change during a paper submission cycle, while research code evolves rapidly.

## Decision
- `src/framework/` — stable, tested, versioned APIs. Changes require an ADR.
- `src/research/` — experimental code for paper contributions. Can change freely.
- `framework/` modules never import from `research/`. `research/` may import from `framework/`.
- Each research project lives in its own subdirectory (e.g. `research/up3renet/`, `research/faupnet/`).

## Consequences
- Framework stability is decoupled from research velocity
- Paper reviewers can inspect stable framework code separately from novel contributions
- Multiple research explorations can coexist without interference

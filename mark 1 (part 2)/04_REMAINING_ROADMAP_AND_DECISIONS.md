# Remaining Roadmap and Decision Gates

## Completed versus remaining

- Completed major steps: `10` through Mark 4E.
- Immediate dataset-characterization tasks remaining before more training: `4` grouped workstreams in one notebook.
- Remaining project phases after the dataset audit: approximately `5` major phases.
- Estimated total workflow completion: approximately `77%`.

## Phase 11 — pre-training dataset characterization

Execute the contract in `03_PRETRAINING_DATASET_AUDIT_CONTRACT.md`.

Decision:

- Critical integrity issue: stop and version/fix data.
- Non-critical documented outliers: freeze the data card and train-derived sampling policy.
- Unsupported sampling hypothesis: retain the existing sampler rather than inventing a new one.

## Phase 12 — fusion freeze and bounded confirmation

Purpose: confirm the selected Mark 4E policy without expanding the search space.

Freeze:

- corrected manifest hash;
- control checkpoint hash;
- recall-loss checkpoint hash;
- broad-window ROI preprocessing;
- pixelwise maximum probability fusion;
- global threshold `0.70`;
- ROI inverse mapping;
- patient and slice metric definitions.

Required checks:

- recompute from checkpoints, not only cached arrays;
- prove cache equivalence within a declared tolerance;
- deterministic repeated inference;
- bootstrap patient uncertainty;
- per-patient and lesion-stratum metrics;
- V104/V116 localization review;
- no additional fusion-weight or threshold search.

Decision:

- Stable full temporary pass: freeze candidate inference policy.
- Failure caused by numerical/caching inconsistency: fix evaluation, do not train.
- Reproducible V116 failure: consider one evidence-driven bounded training refinement.

## Phase 13 — optional single bounded training refinement

Run only if the dataset audit or confirmation provides a predeclared hypothesis. Change one factor at a time. Candidate examples include train-derived hard-positive/domain-balanced sampling or moderate focal alpha; do not choose these automatically.

Required before training:

- frozen data card and sampling JSON;
- train-only bin derivation;
- exact checkpoint initialization;
- epoch and thermal budget;
- stop/resume policy;
- expected-versus-actual validation targets;
- no test access.

If no evidence supports a new intervention, skip this phase and retain the fusion candidate.

## Phase 14 — final inference-policy freeze

Write a single immutable configuration containing:

- dataset/build and manifest hash;
- checkpoints and hashes;
- input generation and ROI rules;
- fusion equation and threshold;
- post-processing;
- metric code version;
- software/runtime details;
- random seeds;
- expected file inventory.

No threshold, checkpoint, fusion or post-processing change is allowed after this freeze based on test results.

## Phase 15 — one-time locked test evaluation

Preconditions:

- dataset gate passed;
- validation confirmation passed;
- inference policy frozen;
- explicit authorization recorded;
- test loader opened once for final evaluation only.

Required test outputs:

- global, positive-patient and per-patient Dice;
- pixel precision/recall;
- lesion-size detection and Dice;
- positive predicted-empty and empty-slice FP;
- uncertainty intervals;
- failure-case panels;
- signed provenance and `test_images_accessed: true` only here.

Test results must not trigger model or threshold tuning.

## Phase 16 — final research package

Produce:

- final technical report/paper draft;
- complete methods and parameter table;
- dataset data card;
- experiment timeline and decision log;
- limitations and failure analysis;
- reproducibility instructions;
- artifact and checksum inventory;
- clear separation of validation-selected and final test results.

## Temporary versus final targets

Mark 4E passed the temporary continuation gate. Older project documentation contains more ambitious final targets, including mean positive-patient Dice around `0.4069`, V104 `0.50`, V116 `0.05`, positive predicted-empty `20%` and empty-slice FP `15%`. Do not claim these final targets were met. Before final freeze, explicitly declare whether they remain formal go/no-go requirements or research aspirations.


# Liver Mark 1 Part 2 — Continuation Workspace

Last verified: 6 August 2026.

This folder is the self-contained handoff for continuing the LiTS liver-tumour segmentation project in a new chat. The earlier work is in `mark 1/`; the next work starts here. A new agent should not require the user to repeat the project history.

## Working-directory rule

This folder is now the exclusive working directory for new phases. Earlier folders are read-only evidence unless the user explicitly requests a historical correction.

Every new activity must receive its own clearly named folder:

`step_NN_short_descriptive_name\`

That folder must contain its runnable `.ipynb`, `README.md`, `outputs\`, configuration/provenance files, error notes and final decision. New outputs must never be written into `mark 1\`, `Practice\`, the project root or another phase's output folder.

## Current decision

- Part 2 Steps 01-21 are complete and the existing scientific evidence chain is sealed.
- The one-time LiTS internal-holdout evaluation is complete; formal model acceptance failed because one positive patient had effectively zero Dice. It must not be rerun or used for tuning.
- The separately frozen 3D-IRCADb-01 evaluation passed its declared contract with patient, small-lesion and negative-control caveats. It must not be reused as a tuning cohort.
- Publication/submission preparation is optional and inactive by owner choice.
- Any future modelling must begin with a genuinely new, predeclared, development-only research question. Existing Step 01 EDA and the canonical dataset provenance record should be reused rather than recomputed.

## Read in this order

1. [AGENTS.md](AGENTS.md) — mandatory instructions for the next agent.
2. [01_PROJECT_AND_DATA_CONTEXT.md](01_PROJECT_AND_DATA_CONTEXT.md) — objective, folders, authoritative dataset, splits and safeguards.
3. [02_COMPLETED_WORK_AND_VERIFIED_RESULTS.md](02_COMPLETED_WORK_AND_VERIFIED_RESULTS.md) — completed steps and authoritative results through Mark 4E.
4. [03_PRETRAINING_DATASET_AUDIT_CONTRACT.md](03_PRETRAINING_DATASET_AUDIT_CONTRACT.md) — exact next notebook, parameters, outputs and gates.
5. [04_REMAINING_ROADMAP_AND_DECISIONS.md](04_REMAINING_ROADMAP_AND_DECISIONS.md) — remaining phases through test and reporting.
6. [05_ARTIFACT_INDEX_AND_REPRODUCIBILITY.md](05_ARTIFACT_INDEX_AND_REPRODUCIBILITY.md) — paths, notebooks, checkpoints, hashes and source-of-truth hierarchy.
7. [06_WORKING_DIRECTORY_AND_DELIVERY_RULES.md](06_WORKING_DIRECTORY_AND_DELIVERY_RULES.md) — mandatory folder, notebook, error-repair and delivery convention.
8. [07_SUCCESS_CRITERIA_PARAMETERS_AND_OUTPUTS.md](07_SUCCESS_CRITERIA_PARAMETERS_AND_OUTPUTS.md) — required parameters, probability artifacts, temporary/final targets and result classification.
9. [08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md](08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md) — canonical LiTS download sources, volume-level source mapping, local evidence, licence caveats and corrected-build identity.

## Canonical dataset-provenance record

Use `08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md` whenever a notebook or report needs the LiTS acquisition history. It supersedes the incomplete shorthand in the historical pipeline notebook and older PNG-oriented project documentation. It does not modify the corrected manifest or any sealed evaluation artifact.

## Status at a glance

| Area | Status |
|---|---|
| Corrected manifest and provenance | Passed |
| Patient-disjoint split audit | Passed |
| Loader, mask and geometry checks | Passed |
| Deterministic ROI overfit | Passed |
| Detailed pre-training dataset characterization | Passed; frozen data card and sampling policy |
| Frozen fusion and inference policy | Passed and sealed |
| One-time LiTS internal-holdout evaluation | Complete; formal model acceptance failed |
| 3D-IRCADb-01 external evaluation | Complete; frozen contract passed with caveats |
| Terminal evidence chain | Verified through Step 21 |
| Dataset acquisition provenance | Consolidated in document 08 |
| Publication workflow | Optional and inactive |
| Further modelling | Requires a new train-only experiment contract |

The declared Part 2 baseline workflow is complete. Completion does not mean that the model passed every scientific acceptance guardrail.

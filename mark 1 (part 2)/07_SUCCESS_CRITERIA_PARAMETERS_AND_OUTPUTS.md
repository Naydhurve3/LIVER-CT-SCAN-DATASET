# Success Criteria, Required Parameters and Output Contract

## Purpose

This document tells a new agent how to decide whether an activity succeeded, what values must be recorded, which probability artifacts are required and what constitutes the final result. It prevents a diagnostic improvement or temporary gate from being described as final project success.

## Result levels

Every phase must declare exactly one result level.

| Level | Meaning | Permitted next action |
|---|---|---|
| `DIAGNOSTIC_COMPLETE` | The question was answered, but no model gate was passed | Plan one evidence-based diagnostic or intervention |
| `FAILED_GATE` | One or more declared mandatory targets failed | Do not advance to test; diagnose or stop |
| `PARTIAL_PASS` | Most targets passed but at least one mandatory target failed | Continue validation-only work |
| `TEMPORARY_CONTINUATION_PASS` | All temporary targets passed | Allow bounded validation confirmation only |
| `VALIDATION_FREEZE_PASS` | Frozen policy reproduced and all declared validation-freeze targets passed | Freeze inference policy and request final test authorization |
| `FINAL_TEST_COMPLETE` | One-time frozen test evaluation completed | Report results; no test-driven tuning |
| `FINAL_PROJECT_COMPLETE` | Test evaluation, uncertainty, failure analysis and reproducibility package are complete | Final paper/report handoff |

The current Mark 4E result is `TEMPORARY_CONTINUATION_PASS`, not final success.

## Current frozen candidate

- Control checkpoint: `mark 1\mark_4_outputs\mark_4_best.pth`.
- Recall checkpoint: `mark 1\mark_4c_outputs\recall_loss_best.pth`.
- Input: one-channel broad CT window derived using `[-160,240]` HU.
- ROI: frozen predicted-liver ROI rule from Mark 3/4.
- Fusion: pixelwise maximum.
- Formula: `p_fused(x) = max(p_control(x), p_recall(x))`.
- Hard threshold: `0.70`.
- Metric population: nine tumour-positive validation patients for mean patient Dice.
- Test state: locked.

Before freeze confirmation, recompute the checkpoint SHA-256 values from disk and save them. Do not rely only on filenames.

## Required parameters for every notebook

Every notebook must show and save these fields when applicable:

### Dataset and data selection

- project root and phase output directory;
- dataset build ID and absolute build path;
- manifest path and SHA-256;
- allowed splits;
- row, slice and patient counts used;
- positive-patient IDs/count;
- excluded rows/patients and reasons;
- test-lock state;
- random seed.

### Input and preprocessing

- source type: NIfTI or derived PNG;
- HU windows and clipping bounds;
- normalization formula and statistics source;
- resize dimensions and interpolation;
- channels and their order;
- ROI source, threshold, component rule, padding and resize;
- augmentation name, probability and range;
- orientation transform handling;
- inverse mapping rule.

### Model and checkpoint

- architecture and input/output channels;
- initialization source;
- checkpoint paths and SHA-256;
- trainable/frozen modules;
- parameter count when training changes;
- device, precision and software versions.

### Training, when applicable

- loss name and all coefficients;
- sampler and exact weights/caps;
- batch size and epoch budget;
- optimizer, learning rate and weight decay;
- scheduler and minimum learning rate;
- gradient clipping;
- batch-normalization policy;
- early-stop/checkpoint selection rule;
- resume state and thermal limits.

### Inference and probability processing

- sigmoid/softmax definition;
- probability dtype and shape;
- fusion equation and weights;
- global threshold grid or frozen threshold;
- post-processing and component filters;
- whether evaluation uses raw, ROI-mapped, liver-gated or fused probabilities.

### Metrics and uncertainty

- exact metric formulas and epsilon;
- patient aggregation population;
- lesion-size definition/bin edges;
- positive-empty and empty-FP denominators;
- bootstrap seed, iterations and resampling unit;
- temporary, freeze and final targets used.

## Required probability artifacts

Any inference, calibration, fusion or confirmation phase must save enough information to reproduce hard predictions without rerunning the model.

### Per-volume probability cache

Use one `.npz` per patient/volume. Required arrays:

| Key | Required content |
|---|---|
| `probability` | Full-image probability array shaped `[slices, height, width]` |
| `truth` | Binary tumour truth aligned to probability |
| `slice_index` | Original axial slice indices |
| `sample_id` | Manifest sample IDs in the same order |
| `volume_id` | Volume identifier, scalar or repeated |

For fusion confirmation, preserve separate `control_probability`, `recall_probability` and `fused_probability`, or preserve the two source caches plus an exact fusion equation.

### Probability integrity checks

- finite values only;
- values in `[0,1]`;
- expected patient and slice coverage;
- no duplicated or missing `sample_id`;
- identical truth/slice ordering between fused checkpoints;
- ROI-to-full-image geometry verified;
- deterministic repeated inference within declared tolerance;
- cache-versus-fresh-inference maximum and mean absolute error.

### Probability diagnostic tables

Save per slice and patient:

- truth pixels and predicted pixels;
- intersection, false-positive and false-negative pixels;
- maximum whole-image probability;
- maximum and median truth-region probability;
- non-tumour-liver probability summary where available;
- detected/empty flags;
- lesion-size stratum;
- patient Dice and slice error rates.

### Probability visualizations

- calibration/threshold dashboard;
- probability histograms for truth, non-tumour liver and background;
- patient-versus-threshold heatmap;
- V104 and V116 CT/truth/probability/prediction/error panels;
- selected-policy patient heatmap;
- reliability/calibration curve when probabilities are described as calibrated.

Do not call sigmoid scores calibrated probabilities unless calibration error or reliability has been assessed. Otherwise call them model probability scores.

## Temporary continuation targets

These were used by Mark 4E and define only a continuation pass.

| Metric | Direction | Temporary target | Mark 4E selected result |
|---|---:|---:|---:|
| Mean Dice over nine positive patients | Higher | `>=0.3329` | `0.377087` |
| V104 Dice | Higher | `>=0.05` | `0.116627` |
| V116 Dice | Higher | `>=0.01` | `0.010474` |
| Q1 detection | Higher | `>=35%` | `50.57%` |
| Positive predicted-empty | Lower | `<=35%` | `27.45%` |
| Empty-slice false positives | Lower | `<=20%` | `5.55%` |

All passed, so Mark 4E is a `TEMPORARY_CONTINUATION_PASS`. The V116 margin is narrow and must be reproduced.

## Validation-freeze requirements

Before test authorization, the agent must produce a confirmation notebook with no new fusion-weight or threshold search. Required conditions:

1. Fresh inference matches the saved caches within a declared tolerance.
2. Pixelwise maximum fusion and threshold `0.70` are used unchanged.
3. All six temporary targets pass again.
4. Patient bootstrap uncertainty is reported.
5. V104 and V116 localization is reviewed.
6. Dataset characterization gate passes.
7. Preprocessing, checkpoints, hashes, fusion, threshold, post-processing and metrics are written to an immutable freeze JSON.
8. `test_images_accessed` remains false.

Recommended strict robustness condition: require the lower bootstrap confidence bound and threshold-neighbour sensitivity to be reported, but do not invent a pass threshold after seeing the results. Declare any additional freeze target before executing confirmation.

## Final success targets

Two target sets exist and must not be confused.

### Formal minimum final acceptance

Before the one-time test run, the project owner/agent must explicitly freeze the final acceptance table. At minimum it should retain the six temporary safety and patient guardrails, require deterministic reproducibility and prohibit catastrophic patient failure.

### Historical aspirational research targets

Older project documentation recorded these stronger goals:

| Metric | Aspirational target |
|---|---:|
| Mean positive-patient Dice | `>=0.406915` |
| V104 Dice | `>=0.50` |
| V116 Dice | `>=0.05` |
| Q1 detection | `>=45%` |
| Positive predicted-empty | `<=20%` |
| Empty-slice false positives | `<=15%` |

Mark 4E does not meet all aspirational targets. These values are research goals unless the user explicitly adopts them as mandatory final gates. The agent must ask for or propose a frozen final acceptance table before test authorization, not after viewing test results.

## Dataset-audit success requirements

Step 01 succeeds only when:

- manifest identity and split integrity pass;
- no critical train-validation leakage is found;
- geometry, affine and label checks have no unresolved critical failure;
- train/validation acquisition, morphology and HU distributions are documented;
- V104/V116 phenotypes and train analog coverage are reported;
- train-derived lesion/sampling bins are saved;
- `DATASET_DATA_CARD.md` exists;
- `sampling_policy.json` exists;
- every required CSV/PNG/JSON in the audit contract exists;
- `pretraining_dataset_gate.json` declares one allowed decision;
- test access remains false.

## Required output files by phase

Every analytical/model phase should produce these standard files in its own `outputs\` directory:

- `configuration.json` — all explicit parameters;
- `provenance.json` — sources, hashes, counts, software and test state;
- `expected_vs_actual.csv` — every gate metric, target, direction and pass flag;
- `gate_result.json` — result level, selected configuration, metrics, decision and next step;
- `patient_metrics.csv`;
- `slice_metrics.csv` when slice inference occurs;
- `lesion_or_size_metrics.csv` when tumour labels are analysed;
- at least one decision dashboard PNG;
- focus-patient localization panels when modelling occurs;
- probability cache and probability diagnostics when inference occurs.

Training phases additionally require:

- `history.csv`;
- `best_checkpoint.pth` and `last_checkpoint.pth` when resumable;
- checkpoint hashes;
- sampler audit;
- gradient/loss-finiteness evidence;
- runtime/temperature record when relevant.

## Machine-readable gate schema

Every `gate_result.json` should contain at least:

```json
{
  "status": "phase_specific_status",
  "result_level": "DIAGNOSTIC_COMPLETE|FAILED_GATE|PARTIAL_PASS|TEMPORARY_CONTINUATION_PASS|VALIDATION_FREEZE_PASS|FINAL_TEST_COMPLETE|FINAL_PROJECT_COMPLETE",
  "selected_configuration": {},
  "selected_metrics": {},
  "targets": {},
  "target_passes": {},
  "all_mandatory_targets_passed": false,
  "decision": "explicit_next_action",
  "next_step": "next_phase_or_stop",
  "manifest_sha256": "...",
  "input_artifact_hashes": {},
  "test_images_accessed": false
}
```

## How the agent should report success

The final message for a phase must state:

1. result level;
2. whether every mandatory target passed;
3. selected parameters/policy;
4. exact key metrics and uncertainty;
5. important failure cases;
6. notebook and output paths;
7. execution/error status;
8. whether test data was accessed;
9. the single next action.

Avoid vague statements such as “the model worked” or “results look good.” Use the formal result level and evidence.


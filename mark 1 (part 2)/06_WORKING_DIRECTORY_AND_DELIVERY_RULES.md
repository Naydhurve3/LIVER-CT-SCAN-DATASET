# Working Directory and Delivery Rules

## Purpose

This contract ensures that every new request produces an organized, runnable and self-contained result under the Part 2 workspace. It also defines how an agent should think, act, react to errors and hand work back to the user.

## Root directory

All new work must be written below:

`D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1 (part 2)`

Older project folders are evidence inputs. Do not write new results into them.

## One activity, one folder

Before creating a notebook, assign the next sequential step number and a concise descriptive name:

`step_NN_short_descriptive_name`

Examples:

- `step_01_pretraining_dataset_characterization`
- `step_02_fusion_freeze_confirmation`
- `step_03_targeted_training_ablation`
- `step_04_final_inference_freeze`

Do not reuse a folder for a scientifically different question. A correction to the same phase stays in that phase and is recorded in `error_fixes.md`.

## Required phase contents

Each activity folder should contain:

| Item | Purpose |
|---|---|
| `README.md` | Question, rationale, inputs, exact parameters, outputs, gate and execution instructions |
| `<folder_name>.ipynb` | Directly runnable notebook |
| `outputs\` | Every generated CSV, JSON, PNG, cache, log or checkpoint |
| `create_*.py` | Optional reproducible notebook generator |
| `patch_*.py` | Optional durable notebook repair |
| `error_fixes.md` | Traceback, cause, repair and rerun instruction when errors occur |

Notebook output constants must resolve to the activity's own `outputs\` directory using an explicit absolute project path or a reliably resolved notebook-relative path.

## Agent operating behavior

The agent is expected to:

- understand the user's requested outcome from context;
- inspect relevant evidence before forming a conclusion;
- think through scientific validity, leakage and reproducibility;
- plan the next bounded step and its gate;
- create the runnable notebook or concrete solution;
- validate structure, syntax, paths and representative execution;
- analyse results when they exist;
- react to errors by finding and repairing the root cause;
- preserve completed expensive work;
- track progress and recommend the next action.

The agent should not stop at generic advice when it can safely create or repair the requested artifact.

## Notebook delivery requirements

Before handing a notebook to the user:

1. Validate it with `nbformat`.
2. Parse every Python code cell.
3. Verify all required input paths.
4. Verify the new output folder is the only write target.
5. Run a safe preflight of setup, provenance, loader and representative model/data operations.
6. State any unexecuted expensive section clearly.
7. Provide a clickable absolute path and tell the user to use Restart Kernel and Run All when appropriate.

## Error-response requirements

When the user provides a traceback:

1. Inspect the actual notebook cell and the real file/schema involved.
2. Explain the verified root cause briefly.
3. Patch the notebook, not merely the displayed snippet.
4. Patch the generator/template if one exists.
5. Reuse saved checkpoints, probability caches and histories.
6. Execute the corrected failing path or its smallest valid preflight.
7. Record the correction in `error_fixes.md`.
8. Tell the user exactly where to resume without unnecessary recomputation.

## Progress tracking

Maintain `PROJECT_STATUS.md` in the Part 2 root. After each completed activity record:

- date and step;
- notebook and output directory;
- executed/error status;
- verified result and gate;
- files created or repaired;
- current completion estimate;
- next recommended step;
- test-lock state.

## Safety boundary

Autonomy does not authorize destructive changes, test-data access, external publication or scientifically different experiments. Ask before crossing those boundaries. Within the requested phase, routine inspection, planning, notebook creation, repair and safe validation should proceed without repeated confirmation.


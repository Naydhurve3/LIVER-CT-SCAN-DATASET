# Error fixes

## Repair 1 - normalized path base

- Symptom: the cohort-freeze cell failed its all-paths-exist assertion.
- Root cause: Step 14 stores `outputs/normalized/...` paths relative to the Step 14 directory, but the first Step 15 draft resolved them relative to the Part 2 root.
- Repair: introduced `STEP14_DIR` and resolve every normalized path against that phase directory.
- Preserved outputs: all Step 14 NIfTI files and completed outputs were untouched.
- Resume: regenerate `step_15.ipynb`, then Restart Kernel and Run All from the first cell.

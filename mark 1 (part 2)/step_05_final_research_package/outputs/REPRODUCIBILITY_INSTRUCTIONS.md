# Reproducibility Instructions

## Controlling artifacts

- Manifest SHA-256: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- Frozen policy SHA-256: `52d2df856709254992ab6b59f84a213a0a00f6e67293f66665b6f258d01f2131`
- Final acceptance contract SHA-256: `12e8932d71651fd9247d841a86dfcd0fa80d64d74c4f334e29b113c28213b52d`
- One-time test run UUID: `871d289b-bf6b-4346-978f-2df02ade26ab`
- Step 04 evidence inventory SHA-256: `9b61a09119128c7b85689762aee51617ee791a2a2b1aafc56adf185dbe6672ab`

## Reproduce the report package

Run `step_05_final_research_package.ipynb` from a fresh Python kernel. It reads only sealed Part 2 outputs, verifies every signed Step 04 evidence hash, recomputes summary metrics from saved tables, and regenerates reports and figures under this phase's `outputs/` folder.

## Do not reproduce the test inference

Step 04 is a completed one-time evaluation. Its ledger has `rerun_allowed: false`. Do not rerun Step 04, rebuild test ROIs, reload test source images/masks, change the frozen policy, or use the test cohort for further selection.

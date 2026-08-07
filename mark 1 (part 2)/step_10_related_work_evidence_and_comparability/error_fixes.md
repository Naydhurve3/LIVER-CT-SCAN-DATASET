# Error fixes

## 5 August 2026 — Gate-label clarification

- Symptom: the gate target key `external quantitative rankings allowed` had value `true`. The value correctly meant that the zero-ranking requirement passed, but the combined key/value could be misread as permission to rank studies.
- Root cause: the expected-versus-actual requirement was phrased as the measured quantity rather than the intended prohibition.
- Repair: changed the requirement to `external quantitative rankings prohibited` in `create_step_10_notebook.py`, regenerated the notebook, and reran the full lightweight artifact-only phase.
- Preserved work: the literature records, manuscript inputs and all source evidence were unchanged; only regenerated Step 10 outputs and their signature were refreshed.
- Resume: the repaired notebook is fully executed. For a manual replay, use **Restart Kernel and Run All** from Cell 1.

## 5 August 2026 — Output-count documentation correction

- Symptom: the README and project status said 19 output files, while the verified phase directory contains 18.
- Root cause: the handoff count included the output package as a conceptual item rather than counting physical files.
- Repair: corrected both Markdown records to 18. No notebook code or generated evidence changed.
- Resume: no rerun is required; the executed notebook and signature remain valid.

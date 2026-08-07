# Step 08 — Venue Selection and Owner Intake

This manuscript-only decision-support phase compares five plausible publication venues using an as-of snapshot of official publisher information, transparent editable weights, explicit editorial-risk notes, and a fail-closed owner intake gate.

## Boundary

The notebook reads only sealed Part 2 manuscript artifacts. It does not access dataset sources or the locked test split, instantiate models or loaders, run inference, alter the frozen policy, or create new scientific results.

## Run

Open `step_08_venue_selection_and_owner_intake.ipynb` in `ds_gpu` and use **Restart Kernel and Run All**. The default owner fields are unresolved, so the notebook completes successfully with `submission_ready: false`.

To record a decision later, edit only the `OWNER_INPUTS` dictionary near the top, then Restart Kernel and Run All. Publisher requirements, policies and fees must be rechecked immediately before submission.

## Interpretation

Scores are transparent decision-support judgments, not acceptance probabilities. A preliminary first-ranked venue is not an owner selection and does not override missing authorship, ethics, dataset-terms, funding, conflict, contribution or availability declarations.

## Verified completion — 5 August 2026

- Result level: `VENUE_DECISION_SUPPORT_COMPLETE`; all 11/11 requirements passed.
- Step 07 prerequisite checks: 8/8 passed.
- Compared five candidates from official publisher pages using six explicit weighted dimensions.
- Preliminary first rank: BMC Medical Imaging, 4.05/5. This is a decision-support result, not an acceptance prediction or owner selection.
- Shortlist: BMC Medical Imaging; Biomedical Signal Processing and Control; Medical Image Analysis as a higher-risk stretch candidate.
- Owner-selected venue: none. Unresolved owner fields: 12. Submission blockers: 19.
- Submission ready: false; the gate correctly failed closed.
- Validation: `nbformat` valid, all six code cells parse and executed successfully with zero errors, the comparison figure was visually inspected, and signed hashes were verified.
- Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Next action: fill `OWNER_INPUTS`, confirm a venue and declarations, then rerun before creating a venue-specific finalization notebook.

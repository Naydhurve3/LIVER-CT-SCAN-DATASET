# Step 10 — Related-Work Evidence and Comparability

This phase converts a primary-literature snapshot into a traceable related-work evidence matrix, comparability audit, citation plan, expanded BibTeX file, venue-neutral Related Work section, and expanded manuscript copy.

## Boundary

The notebook is artifact-only. It reads the sealed Step 09 gate, signature, and scientifically revised manuscript. It does not access dataset sources, test images, masks, probabilities, models, loaders, checkpoints, or inference code. The completed one-time test remains sealed.

## Run

Open `step_10_related_work_evidence_and_comparability.ipynb` in the `ds_gpu` kernel and use **Restart Kernel and Run All**. The embedded source snapshot makes execution offline and deterministic. Every generated artifact is written under `outputs/`.

## Gate

`RELATED_WORK_EVIDENCE_COMPLETE` requires at least 12 verified primary sources, at least five direct LiTS sources, complete evidence-theme coverage, an explicit five-dimension comparability audit, an expanded manuscript and references, and zero external quantitative rankings. It does not mean the manuscript is submission-ready.

## Interpretation

Cross-study Dice values are retained only with their original aggregation and cohort context. No external study shares the exact corrected split, endpoint population, frozen ROI/fusion policy, post-processing, and inclusion rules, so the package prohibits leaderboard, superiority, state-of-the-art, clinical-validity, and deployment claims.

## Next action

After this notebook passes, the owner must complete declarations, verify the exact terms applying to the acquired LiTS copy, and select a venue in Step 08. Venue-specific finalization remains blocked until those inputs are supplied.

## Verified completion — 5 August 2026

- Result level: `RELATED_WORK_EVIDENCE_COMPLETE`; all 11/11 mandatory targets passed.
- Verified the sealed Step 09 boundary and every signed Step 09 artifact before synthesis.
- Curated 13 primary sources, including five directly concerning LiTS, across 13 method, evaluation and reporting themes.
- Created a 14-row comparison matrix containing the current study plus 13 external sources and a five-dimension comparability audit.
- No external source matched the exact corrected split, endpoint population, frozen ROI/fusion policy, post-processing and inclusion rules. Quantitative ranking is therefore disabled for all 13 sources.
- Generated the Related Work section, expanded BibTeX, citation plan, comparability boundaries and `PAPER_DRAFT_WITH_EXPANDED_RELATED_WORK.md`.
- Validation: `nbformat` valid, all six code cells parse, all six executed with zero errors, all 18 output artifacts exist, and the literature evidence map was visually inspected.
- Formal model acceptance remains failed; submission readiness remains false; no venue has been selected.
- Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Next action: complete Step 08 owner inputs, confirm exact dataset terms and select a venue. Only then create venue-specific finalization files.

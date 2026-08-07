# Step 20 — External citation verification and owner readiness

## Purpose

Verify the exact 3D-IRCADb-01 citation and licence from the official IRCAD page, patch a new manuscript copy, and identify the remaining owner-only submission blockers.

## Primary source

- Official IRCAD page: https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/
- Checked: 2026-08-06
- Licence: CC BY-NC-ND 4.0 International
- Official reference: Soler et al., IRCAD technical report, 2010

## Scope

This phase reads only Step 19 signed outputs, the Step 11 owner-status table, and the Step 13 local terms snapshot. It performs no inference, data download, test access, tuning, submission, payment, or owner certification.

## Expected gate

`EXTERNAL_CITATION_VERIFIED_OWNER_DECLARATIONS_REQUIRED`

## Run

Open `step_20.ipynb`, select the `ds_gpu` kernel, restart, and Run All. Every generated artifact is saved under `outputs/`.

## Completed result — 6 August 2026

- Result: `EXTERNAL_CITATION_VERIFIED_OWNER_DECLARATIONS_REQUIRED`; all six gate requirements passed.
- Verified the exact official reference: Soler et al., “3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database,” IRCAD technical report, 2010.
- Verified from the official IRCAD page: 20 patients, 10 women and 10 men, hepatic tumours in 75% of cases, and CC BY-NC-ND 4.0 International licensing.
- The official page supplies no DOI; none was invented.
- Created `Soler2010IRCADb` BibTeX, a citation-patched copy of the Step 19 manuscript, primary-source evidence CSV/JSON, advisory attribution/data-availability drafts, patch plan, and owner blocker matrix.
- Owner status remains 0/14 complete. The notebook did not populate or certify owner-specific declarations.
- Submission, payment, download, inference, tuning, and test access all remained false.
- Combined Step 20 signature: `316c298dcf6e4f027de9d039ce7c2aa8637bc6e37971613bb1b46819008a751d`.

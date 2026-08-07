# Step 07 — Manuscript Evidence and Declaration Scaffold

This venue-agnostic, manuscript-only phase converts the verified Step 06 blockers into a reproducible evidence dossier. It adds primary-source citation candidates, a citation insertion plan, a CLAIM-2024-aligned internal gap map, an owner-fillable declaration template, a dataset-terms verification record, and a citation-keyed manuscript copy.

## Boundary

The notebook reads only sealed Part 2 reporting artifacts. It does not load models or data loaders, access test images or masks, run inference, change the frozen policy, or reopen the locked test split. External source metadata is embedded, so **Run All** requires no network access.

## Run

Open `step_07_manuscript_evidence_and_declaration_scaffold.ipynb` in the `ds_gpu` environment and use **Restart Kernel and Run All**. Every generated artifact is written under `outputs/`.

## Expected interpretation

`MANUSCRIPT_EVIDENCE_SCAFFOLD_COMPLETE` means the evidence and declaration scaffold was created. It does not mean the paper is submission-ready. Owner declarations, authoritative dataset terms and target-venue requirements cannot be inferred and remain mandatory inputs.

## Verified completion — 5 August 2026

- Result level: `MANUSCRIPT_EVIDENCE_SCAFFOLD_COMPLETE`; all 10/10 scaffold requirements passed.
- Sealed inputs: 10/10 checks passed.
- Added four verified primary-source candidates: LiTS, U-Net, MobileNetV2 and CLAIM 2024.
- Created five citation insertion points, a BibTeX file, a citation-keyed manuscript copy, a 16-area reporting gap map, owner metadata and declaration templates, and a dataset-terms verification record.
- Reporting areas supported by existing project evidence: 10/16.
- Unresolved owner or venue metadata fields: 13.
- Submission ready: false. Dataset terms remain explicitly unresolved rather than inferred.
- Validation: `nbformat` valid, all six code cells parse, all six executed successfully with zero cell errors, and the readiness figure was visually inspected.
- Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Next action: the owner completes declarations, verifies the terms applying to the acquired LiTS copy, and selects a target venue before venue-specific finalization.

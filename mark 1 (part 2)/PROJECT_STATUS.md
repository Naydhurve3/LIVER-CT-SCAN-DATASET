# Part 2 Project Status

Last updated: 6 August 2026.

## Current state

- Previous work completed through Mark 4E plus Part 2 Steps 01–05.
- Dataset characterization, bounded fusion confirmation, final inference-policy freeze, one-time test evaluation, and final research package are complete.
- Workflow completion: 100% for the declared Part 2 research contract.
- Current selected validation policy: pixelwise maximum of control and recall-loss probabilities at threshold 0.70.
- Step 02 validation-freeze gate: 6/6 targets passed.
- Formal current project result level: `FINAL_PROJECT_COMPLETE`.
- Final test result level: `FINAL_TEST_COMPLETE`; formal model acceptance failed because V121 did not meet the minimum positive-patient Dice floor.
- Test split: accessed exactly once under run UUID `871d289b-bf6b-4346-978f-2df02ade26ab`; ledger is `COMPLETE` and rerun is prohibited.

## Active next step

- The owner has chosen independent technical research rather than publication/submission preparation. Step 11 venue and author fields are optional and inactive.
- Preserve Steps 01-21 as the completed baseline. Any new modelling phase must use a newly predeclared research question and development-only data; the sealed LiTS holdout and completed external evaluation cannot be reused for tuning.
- The immediate reusable reference is `08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md`, followed by an optional new train-only experiment contract if further model development is desired.
- Do not rerun Step 04 or Step 16, reopen sealed evaluation source data, or use their cohorts for model, threshold, fusion, ROI, post-processing or training decisions.

## Step 01 creation update — 3 August 2026

- Created the directly runnable Step 01 notebook and durable generator under the phase folder.
- Implemented all contract sections: manifest/integrity, geometry, 3D lesion morphology, HU/domain shift, leakage, label/ROI QC, V104/V116 train-analog analysis, difficulty associations, data card, sampling policy, and machine-readable gates.
- Verified the manifest SHA-256 from disk and reconciled its 58,638-row, 27-column schema and split counts without dereferencing test paths.
- Validation status: `nbformat` valid, every code cell parses, and safe setup/manifest/test-lock preflight passes.
- Full execution status: not executed by design; expensive train/validation NIfTI analysis awaits manual Restart Kernel and Run All.
- Current project result remains `TEMPORARY_CONTINUATION_PASS` from Mark 4E. Step 01 has no result level until its outputs execute and reconcile.
- Repairs made during notebook generation are recorded in the phase `error_fixes.md`.
- Test split: locked; test images, masks, and statistics were not accessed.
- Single next action: manually Run All in Step 01; if the dataset gate passes, proceed to Step 02 bounded fusion confirmation.

## Step 01 repair update — 3 August 2026

- Repaired Cell 6 after the active Matplotlib API rejected `Axes.boxplot(labels=...)`.
- Updated both the runnable notebook and `create_step_01_notebook.py` to use `tick_labels=ALLOWED_SPLITS`.
- Preserved the completed integrity, geometry, lesion, HU, label-QC, and ROI CSV outputs; the expensive source-NIfTI profiling does not need to be repeated while its in-memory variables remain available.
- Post-repair validation: notebook is `nbformat` valid, all 11 code cells parse, and the active environment's Matplotlib source declares `tick_labels` for `Axes.boxplot`.
- Resume point: rerun Cell 6, then continue with Cell 7 onward. If the kernel state was lost, Restart Kernel and Run All is required to recreate the in-memory geometry tables.
- Test split remains locked and was not accessed during repair.

## Step 01 audit-validity repair — 3 August 2026

- The first completed Run All produced `FAILED_GATE / HOLD_FIX_DATA_OR_LABEL_ISSUE`, but review showed that result was caused by notebook logic defects and is superseded.
- Tumour-empty volumes were incorrectly assigned zero containment instead of not-applicable containment.
- The approved manifest `rot180` transform was not applied to source segmentation before HU/image-label analysis, invalidating affected HU/domain results.
- The notebook and generator now apply the approved transform, compose it into the effective affine, distinguish raw/effective/resolved affine state, and gate only finite tumour containment.
- Preflight evidence: all 33 rotated train/validation volumes numerically align after transform composition; all 117 train/validation volumes have resolved alignment. Volumes 48–52 retain manually approved header inconsistency and remain review cases, not automatic critical failures.
- Current Step 01 state: `REPAIRED_RERUN_REQUIRED`; the existing gate, data card, HU/appearance, difficulty, and downstream decision artifacts must not be used for advancement.
- Test split remains locked and was not accessed.
- Single next action: Restart Kernel and Run All in the repaired Step 01 notebook, then re-review the regenerated machine-readable gate before planning Step 02.

## Step 01 completion — 3 August 2026

- Repaired notebook completed at approximately 14:22 local time and regenerated every required artifact.
- Final result level: `DIAGNOSTIC_COMPLETE`.
- Mandatory targets: 12/12 passed; all 36 required outputs exist.
- Gate decision: `PASS_FREEZE_DATA_CARD_AND_PROCEED_TO_FUSION_CONFIRMATION`.
- Critical failures: zero integrity, leakage, geometry, label, and frozen-ROI failures.
- Alignment: 117/117 volumes resolved; the five remaining raw-header mismatches are manually approved train volumes 48–52 and are documented review cases.
- Validation is enriched for smaller lesions, but every train-derived lesion-volume, diameter, and patient-burden stratum has validation coverage.
- V104 is a multifocal, very-low-contrast phenotype; V116 is a large solitary, weak-contrast phenotype. Both have train analogs, and neither failure is explained by ROI clipping.
- Frozen sampling decision: `uniform_patient_aware_existing_sampler_no_change` with train-derived bins and seed 42.
- Test split: locked; `test_images_accessed: false`.
- Single next action: create and run Step 02 bounded fusion freeze confirmation with maximum fusion and global threshold 0.70 unchanged.

## Step 02 creation — 3 August 2026

- Created `step_02_fusion_freeze_confirmation/step_02_fusion_freeze_confirmation.ipynb` and its durable generator.
- Mode: validation confirmation only; no training and no policy search.
- Frozen candidate: control plus recall-loss checkpoints, predicted-liver ROI threshold 0.50/largest-3D/padding 16, pixelwise maximum fusion, global threshold 0.70, no post-processing.
- Mandatory checks include fresh-versus-historical cache equivalence, two-pass deterministic inference, all six temporary targets, patient bootstrap uncertainty, component/size diagnostics, score reliability, and V104/V116 localization.
- Threshold 0.65 is declared report-only sensitivity and cannot replace threshold 0.70.
- Safe preflight passed: Step 01 gate, manifest hash, both checkpoint payloads/hashes, 13 validation ROI/cache pairs, 10,685 validation rows, nine positive patients, one representative ROI sample, and sealed test state.
- Full execution status: not run by the agent; the notebook performs four full validation inference passes and awaits manual Restart Kernel and Run All.
- Test split: locked; `test_images_accessed: false`.
- Single next action: manually Run All Step 02 and review `outputs/gate_result.json`.

## Step 02 cache-equivalence repair — 3 August 2026

- The first execution completed all 13 volumes for both checkpoints and both deterministic passes, then stopped at a historical-cache equivalence assertion.
- Determinism passed exactly for all 26 model-volume pairs.
- Historical-cache differences were sparse: mean absolute error approximately `1e-8` to `1.6e-6`; after fusion only 97 of 700,252,160 pixels changed hard class at threshold 0.70 (`1.385e-7`). Maximum patient-Dice change was approximately 0.000189.
- Root cause: a single-pixel maximum-error rule was too brittle for historical float16 cache comparison.
- Patched notebook and generator now gate aggregate mean error, fraction of outlier score pixels, hard-prediction disagreement, and final metric equivalence. Maximum pixel error remains reported as diagnostic evidence.
- All completed fresh caches, determinism checks, cache integrity, and runtime logs were preserved.
- Resume with live kernel: rerun setup/preflight code Cells 1–4, skip expensive code Cell 5, and continue from code Cell 6. If kernel state was lost, Restart Kernel and Run All.
- Test split remains locked and was not accessed.

## Step 02 completion — 3 August 2026

- Final result level: `VALIDATION_FREEZE_PASS`.
- Every mandatory confirmation requirement passed, including Step 01, manifest/checkpoint hashes, deterministic inference, cache and hard-prediction equivalence, all six temporary targets, uncertainty reporting, localization panels, and test lock.
- Confirmed maximum-fusion threshold-0.70 metrics: mean positive-patient Dice 0.37706060; V104 0.11647368; V116 0.01047342; Q1 detection 50.570342%; positive predicted-empty 27.447217%; empty-slice false positives 5.548066%.
- Patient-bootstrap 95% interval for mean Dice: 0.19792987 to 0.55420929 using 10,000 patient resamples.
- Determinism maximum difference: 0.0. Aggregate fresh/historical mean score difference approximately 1.09e-7; hard-prediction disagreement approximately 1.21e-7.
- Important limitations: V116 margin above its temporary target is only about 0.000473; smallest derived component-quartile detection is 13.04%; positive predicted-empty remains 27.45%; historical aspirational targets are not all met.
- Frozen checkpoint hashes: control `9b0c7749af66b0fc3808f6757d0d90af01384e06afe02090361b220df48b6e8b`; recall `c01eb4b81e4e7f1d84c7966aca56e738d87d06d404907f0bcc7c67a79ed4ec4d`.
- Test split remains locked; `test_images_accessed: false`.
- Single next action: create Step 03 final inference-policy freeze and formal final acceptance contract. Do not access test data until Step 03 passes and explicit one-time authorization is recorded.

## Update rule

The agent must append or revise this file after every completed phase or repaired execution. Results must come from executed machine-readable artifacts, not from planned values.

## Step 03 completion — 3 August 2026

- Created and executed `step_03_final_inference_policy_freeze/step_03_final_inference_policy_freeze.ipynb` plus its durable generator, README, output package, and repair log.
- Mode: lightweight artifact and contract verification only; no training, model inference, test loader, test image, test mask, test probability, or test statistic access.
- Final result level remains `VALIDATION_FREEZE_PASS`; all 10/10 policy-freeze readiness requirements passed after completeness repair.
- Reverified Step 01 and Step 02 prerequisite gates, the authoritative manifest hash, both checkpoint hashes, the validation ROI-manifest hash, and all immutable inference-policy fields.
- Checksum inventory contains 38 authoritative artifacts, including all 13 frozen validation probability caches, the liver-ROI generator checkpoint, loader/model source files, prerequisite gates, metrics, configuration, provenance, and the Step 02 notebook.
- Final frozen policy: MobileNetV2UNet control and recall-loss checkpoints; one broad `[-160,240]` HU channel normalized as uint8/255; predicted-liver ROI threshold 0.50, largest-3D component and padding 16; pixelwise maximum fusion; global threshold 0.70; no post-processing; epsilon `1e-6`; seed 42.
- Formal minimum final acceptance was declared before test access. The six validation guardrails remain prerequisites; the final test contract also prohibits catastrophic positive-patient failure and requires complete, finite, unique sample evaluation. Historical stronger targets remain research aspirations rather than mandatory final gates.
- Machine-readable decision: `REQUEST_EXPLICIT_ONE_TIME_TEST_AUTHORIZATION`.
- Authorization state: ready but not granted. No Step 04 test evaluation may start until the owner explicitly authorizes exactly one run using the frozen Step 03 policy and acceptance contract.
- Test split remains locked; `test_images_accessed: false`.
- Single next action: review the Step 03 frozen policy and acceptance contract, then record explicit one-time test authorization before creating or running Step 04.

## Step 03 completeness repair — 3 August 2026

- Step 04 planning found that the initial freeze named a predicted-liver ROI but omitted the generator checkpoint and exact unseen-cohort ROI construction contract.
- Repaired both the Step 03 notebook and generator before test access.
- Added and verified the liver checkpoint/hash, two-output architecture, liver channel 0, robust per-slice normalization, 26-connectivity, threshold 0.50, padding 16, and a predeclared full-image fallback for empty ROI predictions.
- Added the liver checkpoint to the signed artifact inventory.
- Froze the final-test Q1 slice definition at 1–51 tumour pixels, where 51 is the train-only 25th percentile from 4,930 positive training slices.
- Re-executed Step 03 safely: 10/10 readiness requirements passed, 38 artifacts were checksummed, and test access remained false.

## Step 04 creation — 3 August 2026

- Created the directly runnable one-time test notebook and durable generator under `step_04_one_time_locked_test_evaluation_after_explicit_authorization/`.
- Implemented frozen predicted-liver ROI generation, single-pass control/recall inference, maximum fusion at threshold 0.70, per-volume probability caches, global/patient/slice metrics, patient bootstrap uncertainty, train-derived lesion-volume strata, probability diagnostics, failure panels, final acceptance evaluation, signed evidence, and a sealed one-time run ledger.
- Added interruption-safe resume semantics under the same UUID; a completed ledger cannot be rerun.
- Authorization defaults remain locked: `AUTHORIZATION_GRANTED = False`, blank authorization text/timestamp/run UUID, and no test-loader construction before the gate.
- Validation status: `nbformat` valid, all 10 code cells parse, the generator compiles, and both safe-preflight cells executed successfully.
- Safe preflight reverified the manifest and control, recall-loss, and liver-ROI checkpoint hashes against the repaired Step 03 signature.
- Test paths dereferenced: false. Test images, masks, probabilities, and statistics accessed: false.
- Execution status: intentionally not run beyond preflight because explicit one-time authorization has not been granted.
- Single next action: if the frozen policy and acceptance contract are accepted, provide the exact authorization sentence recorded in the Step 04 README; then patch the authorization record and Run All exactly once.

## Step 04 authorization update — 5 August 2026

- The notebook's intentional Cell 3 authorization assertion stopped the first manual attempt before test access.
- The user then repeatedly instructed the agent to make the required authorization edit.
- Patched both the generator and notebook with `AUTHORIZATION_GRANTED = True`, the canonical authorization text, UTC timestamp `2026-08-05T11:39:37.3247843Z`, and one-time run UUID `871d289b-bf6b-4346-978f-2df02ade26ab`.
- No test data was accessed during the patch; the only existing Step 04 output remains the safe-preflight record.
- Single next action: Restart Kernel and Run All exactly once. If interrupted, repair/resume under the same UUID; never start a second run after the ledger reaches `COMPLETE`.

## Step 04 completion — 5 August 2026

- One-time run UUID `871d289b-bf6b-4346-978f-2df02ade26ab` completed and the ledger was sealed with `rerun_allowed: false`.
- Result level: `FINAL_TEST_COMPLETE`.
- Formal acceptance: 7/8 final test/integrity targets passed; overall formal model acceptance failed.
- Test metrics: global Dice 0.767696; mean positive-patient Dice 0.507297 with 95% bootstrap interval 0.344175–0.659411; global precision 0.844394; global recall 0.703771; Q1 detection 47.552%; positive predicted-empty 9.942%; empty-slice false positives 3.810%.
- Failed guardrail: V121 had effectively zero Dice versus the predeclared minimum positive-patient floor of 0.01.
- Decision: `REPORT_FINAL_TEST_GATE_FAILURE_NO_TUNING_NO_RERUN`.

## Step 05 completion — 5 August 2026

- Created and executed `step_05_final_research_package/step_05_final_research_package.ipynb` and its durable generator.
- Mode: report-only, no inference, no loader/model/checkpoint execution, and no test source images or masks reopened.
- Result level: `FINAL_PROJECT_COMPLETE`; 12/12 final-package requirements passed.
- Formal model acceptance remains false and is explicitly preserved in the gate, data card, technical report and paper draft.
- Independently reconciled all headline test metrics from sealed tables within 1e-7 and verified the Step 04 ledger, signature and signed evidence inventory.
- Characterized V121 from the signed cache: 100% ROI containment, no truth-region pixel at or above 0.70, supporting a recognition failure rather than ROI clipping.
- Generated 23 required outputs including the final technical report, internal paper draft, methods table, experiment timeline, decision log, limitations analysis, reproducibility instructions, data card, three publication figures, 34-row checksum inventory and final package signature.
- Validation: `nbformat` valid, all eight code cells parse, zero cell errors, 23/23 outputs present, 34/34 inventory hashes verified, figures visually inspected.
- Final decision: `ARCHIVE_FINAL_RESULT_FORMAL_MODEL_ACCEPTANCE_FAILED_NO_FURTHER_TEST_USE`.
- Only remaining work is manuscript editing, references and dissemination from sealed Step 05 outputs; no further test access is allowed.

## Step 06 completion — 5 August 2026

- Created and validated `step_06_manuscript_submission_readiness_and_archive/step_06_manuscript_submission_readiness_and_archive.ipynb` plus its durable generator, README, repair log and output package.
- Mode: post-project artifact-only audit; no training, model/checkpoint loading, inference, policy change, test loader, test source image, test mask or new test statistic access.
- Result level: `DIAGNOSTIC_COMPLETE`; all 8/8 audit-package requirements passed.
- Submission readiness remains false. The scientific narrative passed 10/10 internal content checks, but 11 manual/external items remain: literature references; authors/affiliations; ethics/applicability; dataset terms; author contributions; conflicts; funding; code/data availability; target venue; venue formatting; and reporting-guideline mapping.
- Verified 19/19 sealed-package and ledger checks and mapped 9/9 major scientific claims to their authoritative evidence.
- Created a 160-file SHA-256 archive manifest covering 15.5 MiB of Part 2 contracts, code, reports, tables, figures and gates. Twenty-seven bulky cache/model files were deliberately not re-hashed; their integrity remains covered by the signed Step 03–05 inventories.
- Validation: `nbformat` valid, all five code cells parse, full artifact-only execution succeeded, all expected outputs exist, and the submission-readiness dashboard was visually inspected.
- Formal model acceptance remains failed because V121 did not meet the frozen minimum positive-patient Dice floor. The one-time Step 04 ledger remains complete with rerun prohibited.
- Test images accessed in Step 06: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: manually complete the manuscript metadata, external references and venue-specific review from the sealed Step 05/06 package. There is no further Part 2 modeling or test-evaluation phase.

## Step 07 completion — 5 August 2026

- Created and fully executed `step_07_manuscript_evidence_and_declaration_scaffold/step_07_manuscript_evidence_and_declaration_scaffold.ipynb` plus its durable generator, README, repair log and 18-output dossier.
- Mode: offline artifact-only manuscript preparation; no model, loader, checkpoint, test image, test mask, probability cache, inference, policy change or new test statistic access.
- Result level: `MANUSCRIPT_EVIDENCE_SCAFFOLD_COMPLETE`; all 10/10 mandatory scaffold requirements and 10/10 sealed-input checks passed.
- Added four traceable primary-source candidates covering LiTS, U-Net, MobileNetV2 and the CLAIM 2024 reporting framework; created five citation insertion points and a BibTeX file.
- Generated a citation-keyed manuscript copy, 16-area reporting-gap map, declaration template, owner metadata template, dataset-terms verification record, source snapshot, completion report, figure, provenance, signature and machine-readable gate.
- Existing project evidence supports 10/16 reporting areas. Thirteen owner or venue fields remain unresolved. Submission readiness remains false.
- The exact license/terms applying to the owner's acquired LiTS copy were not verified and were deliberately not inferred from third-party mirrors. Owner confirmation remains mandatory.
- Formal model acceptance remains failed because of the V121 minimum-patient guardrail. This negative result is preserved in the revised manuscript and gate.
- Validation: `nbformat` valid, all six code cells parse, all six executed successfully with zero errors, required outputs exist, and the readiness figure was visually inspected.
- Test images accessed in Step 07: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: complete owner/institutional declarations, verify dataset terms, and select a target venue. Only then create a venue-specific finalization phase; no further Part 2 modeling or test evaluation is permitted.

## Step 08 completion — 5 August 2026

- Created and fully executed `step_08_venue_selection_and_owner_intake/step_08_venue_selection_and_owner_intake.ipynb` plus its generator, README, error log and 17-output decision dossier.
- Mode: manuscript-only venue decision support. No dataset source, test image, test mask, probability cache, model, loader, checkpoint, inference, policy change or new test statistic was accessed.
- Result level: `VENUE_DECISION_SUPPORT_COMPLETE`; all 11/11 phase requirements and 8/8 Step 07 prerequisite checks passed.
- Compared BMC Medical Imaging, Biomedical Signal Processing and Control, Computers in Biology and Medicine, Medical Image Analysis and Radiology: Artificial Intelligence using official publisher information verified on 5 August 2026.
- Transparent default weights assessed scope fit, evidence compatibility, novelty alignment, reporting alignment, format compatibility and cost flexibility. Scores are decision-support judgments, not acceptance probabilities.
- Preliminary first rank: BMC Medical Imaging at 4.05/5. Biomedical Signal Processing and Control ranked second at 3.60/5. Medical Image Analysis ranked third at 3.275/5 but is explicitly labeled a high-risk stretch candidate.
- No venue was owner-selected. Twelve owner input fields and 19 combined submission blockers remain unresolved; `submission_ready` correctly remains false.
- Current venue instructions, policies and fees must be reverified immediately before submission. The notebook embeds an as-of source snapshot and editable weights/owner inputs.
- Formal model acceptance remains failed because of the V121 minimum-patient guardrail and is preserved in the decision report and gate.
- Validation: `nbformat` valid, all six code cells parse and executed with zero errors, all required outputs exist, the venue figure was visually inspected, and the signed artifact hashes passed.
- Test images accessed in Step 08: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: edit `OWNER_INPUTS` in Step 08 to confirm the selected venue and complete owner/institutional declarations, then Restart Kernel and Run All. Do not create venue-specific submission files until that gate passes.

## Step 09 completion — 5 August 2026

- Step 08 was rechecked after the user's run: the phase remained valid, but no venue or owner fields were supplied. The venue-specific gate therefore remained correctly closed.
- Created and fully executed `step_09_internal_peer_review_and_claim_consistency/step_09_internal_peer_review_and_claim_consistency.ipynb` plus its generator, README, repair log and 20-output internal-review dossier.
- Mode: venue-neutral, artifact-only internal simulated peer review. This is not independent external peer review and did not access dataset sources or the locked test split.
- Result level: `INTERNAL_PEER_REVIEW_COMPLETE`; all 11/11 phase requirements passed. Overall assessment: `SHARE_WITH_CAVEATS`.
- Verified 9/9 sealed inputs, 12/12 displayed numerical claims, 7/7 Step 04-to-Step 05 metric crosschecks, 13/13 frozen-method claims, 9/9 required manuscript sections and 4/4 foundational citation keys.
- Scientific consistency passed. Generated `PAPER_DRAFT_SCIENTIFICALLY_REVISED.md`, which narrows claims about generalization/transfer, removes distracting aspirational comparisons, makes the frozen 16-pixel ROI padding explicit, preserves the negative formal gate, and corrects encoding artifacts without changing scientific results.
- Produced an internal review report, reviewer comments, editorial-risk register, revision checklist, scorecard, reconciliation tables, dashboard, provenance, signature and machine-readable gate.
- Formal model acceptance remains failed because of the V121 minimum-patient floor. External validation remains absent and no clinical, deployment or state-of-the-art claim is supported.
- Submission readiness remains false. The outstanding blockers are venue-grade related-work depth, owner/institutional declarations, exact LiTS dataset terms and owner-confirmed venue selection.
- Validation: `nbformat` valid, all six code cells parse and executed successfully with zero errors, the dashboard was visually inspected, and signed artifact hashes passed.
- Test images accessed in Step 09: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: expand the related-work comparison using current primary literature and complete Step 08 owner inputs. Only after the owner gate passes may venue-specific formatting begin.

## Step 10 completion — 5 August 2026

- Created and fully executed `step_10_related_work_evidence_and_comparability/step_10_related_work_evidence_and_comparability.ipynb` plus its durable generator, README, repair log and 18-output evidence package.
- Mode: offline artifact-only primary-literature synthesis. It read only the sealed Step 09 gate, signature and revised manuscript; it did not access dataset sources or the locked test split.
- Result level: `RELATED_WORK_EVIDENCE_COMPLETE`; all 11/11 mandatory targets and 10/10 sealed-boundary checks passed.
- Curated 13 verified primary sources, including five directly concerning LiTS, and covered 13 architecture, loss, evaluation and reporting themes.
- Produced a comparison matrix for the current study plus 13 external sources, a five-dimension comparability audit, citation insertion plan, expanded BibTeX, venue-neutral Related Work section, claim-boundary record and expanded manuscript.
- No external paper matched the corrected split, metric population, frozen ROI/fusion policy, post-processing and inclusion rules. All 13 external quantitative rankings are prohibited; no Dice leaderboard, superiority or state-of-the-art claim was generated.
- The defensible positioning is a reproducible frozen-fusion held-out evaluation with explicit patient- and lesion-level failure reporting. Formal model acceptance remains failed because of V121.
- Validation: `nbformat` valid, all six code cells parse and executed with zero errors, all 18 outputs exist, the evidence-map figure was visually inspected, and 16 generated evidence artifacts were SHA-256 signed.
- Submission readiness remains false and venue selection remains null. The current blockers are owner/institutional declarations, exact terms for the acquired LiTS copy, and owner-confirmed venue selection.
- Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: complete the Step 08 owner inputs and select the venue. Only after that gate passes should a venue-specific finalization notebook be created; no further modeling or test evaluation is permitted.

## Step 11 creation and safe owner-gate run — 5 August 2026

- Reverified the completed Step 10 rerun: `RELATED_WORK_EVIDENCE_COMPLETE`, 13 primary sources, five LiTS-direct sources, quantitative ranking prohibited, signed hashes valid, and no test access.
- Step 08 remains unresolved: no owner-selected venue and the previous owner-intake gate is not submission-ready.
- Created and executed `step_11_owner_submission_gate_and_finalization_handoff/step_11_owner_submission_gate_and_finalization_handoff.ipynb` plus its durable generator, README, repair log and 13-output owner-action package.
- Mode: artifact-only owner submission gate. It does not infer authorship, ethics/applicability, dataset terms, contributions, conflicts, funding, availability, publication budget or venue choice.
- Current result level: `OWNER_INPUT_REQUIRED`. This is an expected successful notebook state, not an execution failure.
- Phase-integrity requirements: 7/7 passed; sealed-input checks: 12/12 passed. Owner fields complete: 0/14; unresolved: 14; owner gate passed: false; venue selected: none.
- The improved schema requests actual owner statements rather than treating generic boolean flags as sufficient evidence. It validates venue membership, explicit open-access preference, non-negative budget, email syntax, substantive declarations and final owner certification.
- Submission readiness, submission authorization and payment authorization all remain false. Formal model acceptance remains failed because of V121.
- Validation: `nbformat` valid, all five code cells parse and executed with zero errors, 13 outputs exist, all 12 signed hashes pass, and the owner-gate dashboard was visually inspected.
- Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: edit only `OWNER_INPUTS` in Step 11 code Cell 1 and use Restart Kernel and Run All. After `OWNER_INPUT_GATE_PASS`, refresh the selected venue's live instructions and create the final venue-specific formatting package. No external submission or payment is authorized by the notebook.

## Scope correction after Step 11 — 5 August 2026

- The owner clarified that publication/submission preparation is not the project objective; the goal is to continue working with public online data and the data already available locally.
- Step 11 is retained as an optional provenance artifact but is no longer an active blocker or required next phase. Its missing author, venue, ethics, funding and submission fields do not block further data-science work.
- The active direction is external public-data compatibility, ingestion, quality control and—only after a new frozen gate—external evaluation. The sealed LiTS test split remains closed.

## Step 12 completion — 5 August 2026

- Created and fully executed `step_12_public_external_dataset_audit_and_acquisition_plan/step_12_public_external_dataset_audit_and_acquisition_plan.ipynb` plus its generator, README, repair log and 16-output metadata package.
- Mode: online metadata plus existing Step 01 train/validation-derived references only. No online dataset was downloaded and no local test source was accessed.
- Result level: `PUBLIC_EXTERNAL_DATA_AUDIT_COMPLETE`; all 7/7 requirements and 8/8 internal-boundary checks passed.
- Audited four authoritative public sources: 3D-IRCADb-01, HCC-TACE-Seg, MSD Task03 Liver and CHAOS CT.
- Recommended first source: 3D-IRCADb-01 (`4.25/5`) because it is a small independent 20-case CT cohort with hepatic tumours in 75% of cases and source-provided liver/tumour structures.
- Recommended second source: HCC-TACE-Seg (`3.95/5`) because it provides 105 independent HCC subjects, but its approximately 28.57 GB DICOM/DICOM-SEG package, longitudinal phase selection and documented HCC_001 dimension mismatch require a heavier ingestion audit.
- MSD Task03 Liver is prohibited as independent external validation because the MSD publication identifies it as a subset of LiTS patients. CHAOS CT is limited to liver-ROI/domain-shift analysis because its CT cohort contains healthy livers without tumours.
- Created a 12-check acquisition/conversion contract covering terms, hashes, patient identity, series selection, DICOM geometry, DICOM-SEG references, label semantics, conversion parity, HU semantics, overlap leakage, preprocessing and a frozen evaluation policy.
- Repaired one safe-preflight schema mismatch: Step 01 uses split label `val`, not `validation`. No test split was present. Also clarified two machine-readable safeguard labels so successful zero-count checks cannot be misread as download/test access.
- Validation: `nbformat` valid, six code cells parsed and executed, zero errors, 16 outputs exist, all 15 signed hashes pass, and the dashboard was visually inspected.
- Downloads performed: false. Test images accessed: false. Test source files reopened: false. Test inference rerun: false.
- Single next action: confirm the official 3D-IRCADb-01 reuse terms and authorize a bounded Step 13 download/ingestion-and-QC phase. Do not run inference until all 12 QC checks pass.

## Step 12 notebook-open repair — 5 August 2026

- The user reported that the long notebook filename could not be opened.
- Verified that the original file exists, is valid JSON, passes `nbformat`, contains 14 cells and six executed code cells, and is not corrupt.
- The original full path is 184 characters and includes spaces and parentheses; a client-side Windows/Jupyter path-handling failure is the likely cause.
- Created the identical executed short-name alias `step_12_public_external_dataset_audit_and_acquisition_plan/step_12.ipynb` and updated the generator and README to preserve it.
- All 16 Step 12 output artifacts and signed hashes were preserved; no analysis rerun or test access was required.
- Resume: open `step_12.ipynb` in the Step 12 folder.

## Step 13 creation and safe preflight — 5 August 2026

- Reverified the completed Step 12 gate and all 15 signed source-audit artifacts. The recommended source remains 3D-IRCADb-01 and the local LiTS test split remains sealed.
- Refreshed the official IRCAD source page: the combined archive is approximately 782 MB; the 20 cases are available as DICOM images, labelled DICOM, per-structure DICOM masks and VTK meshes; and the page licenses the work under CC BY-NC-ND 4.0 with a required Soler et al. citation.
- Created `step_13_3d_ircadb_ingestion_and_qc/step_13.ipynb` plus the descriptive-name notebook, durable generator, README, repair log and guarded output/staging structure.
- Implemented explicit user terms acceptance, download and extraction switches; streaming temporary-file download; archive size and ZIP validation; ZIP path-traversal protection; archive and extracted-file inventories; DICOM header/geometry profiling; SOP UID uniqueness; label-folder classification; patient-level liver/tumour availability; machine-readable gates; provenance; and signatures.
- Default safe-preflight result: `EXTERNAL_DOWNLOAD_AUTHORIZATION_REQUIRED`. This is expected and not an execution failure.
- Verified 6/6 Step 12 prerequisites and 5/5 zero-event/integrity safeguards. Terms and attribution checks passed; acquisition and source-QC checks remain visibly amber/pending because no archive was downloaded.
- Corrected machine-readable zero-event labels and dashboard semantics so pending acquisition checks cannot be mistaken for observed failures.
- Validation: both notebook filenames pass `nbformat` and AST parsing; all six short-notebook code cells executed with zero errors; 17 output files and two staging directories exist; all 16 signed hashes pass; dashboard visually inspected.
- Download performed: false. Conversion performed: false. Inference performed: false. Test images accessed: false. Test source files reopened: false.
- Single next action: if the official CC BY-NC-ND 4.0 conditions are acceptable, set `USER_ACCEPTS_CC_BY_NC_ND_4_0`, `DOWNLOAD_ENABLED`, and `EXTRACT_ENABLED` to `True` in Step 13 code Cell 1, then Restart Kernel and Run All. Do not proceed to conversion or inference unless the source-ingestion gate passes.

## Step 13 authorized ingestion completion — 5 August 2026

- The user accepted the official 3D-IRCADb-01 CC BY-NC-ND 4.0 terms and authorized bounded download and extraction.
- Downloaded and verified 20 official patient ZIP archives totaling 820,269,486 bytes. Archive-set SHA-256: `55cd6722da6c02f8365ab566603db116128e0d854782d2a174b85056f4a5ba96`.
- Result level: `EXTERNAL_SOURCE_INGESTION_QC_PASS`; 20/20 patients, one readable CT series per case, 74–260 slices, 512 × 512 geometry, all liver labels present, and zero duplicate SOP Instance UIDs.
- Repaired the official patient-level downloader, missing `pydicom` dependency, nested archive extraction, repeated inner DICOM folders, and a rerun regression that counted both `PATIENT_DICOM` levels as 40 patients. The generator, notebook and repair log contain the durable fixes.
- Conversion and inference remained disabled in Step 13. The local LiTS test split was not accessed or reopened.
- Decision: proceed to normalized conversion and source-parity QC.

## Step 14 normalized conversion and parity-QC completion — 5 August 2026

- Created and fully executed `step_14_3d_ircadb_normalized_conversion_and_parity_qc/step_14.ipynb` plus its descriptive-name notebook, durable generator, README and repair log.
- Result level: `EXTERNAL_NORMALIZED_CONVERSION_QC_PASS`; all 12/12 mandatory checks passed and the signed Step 13 prerequisite was independently reverified.
- Converted all 20 external cases to internal compressed NIfTI CT, liver-mask and tumour-mask volumes. Exact source-to-reloaded serialization parity passed for all three arrays in all cases; all CT values were finite; all liver masks were nonempty; and every mask slice matched its CT instance and position.
- Pixel inspection reconciled exactly 15/20 hepatic-tumour cases. The repaired definition includes `livertumor`, `livertumors` and numbered variants while excluding case 5 adrenal tumours and case 7's generic non-liver tumour label.
- External geometry spans 74–260 slices, 0.561–0.873 mm in-plane spacing and 1–4 mm slice spacing. Tumour-to-liver containment is at least 98.545%; 223 source tumour voxels outside liver are preserved and reported rather than clipped.
- Generated normalized volumes, conversion manifest with per-file hashes, geometry/HU/label profiles, tumour-folder reconciliation, containment QC, dashboard, configuration, provenance, machine-readable gate and signature. Final combined Step 14 signature: `a023b54f4b93ca935162824b6e51d3e70a405383ae38ba5b89c675a24839dc90`; it supersedes the earlier time-dependent signature after the repaired final rerun.
- Validation: `nbformat` valid, all code cells parse, the repaired notebook executed with zero errors, all signed artifacts exist, and the gate passed.
- Model inference performed: false. Formal external evaluation performed: false. Local LiTS test images, masks, statistics and loaders accessed: false.
- Single next action: create Step 15 to freeze the external-evaluation cohort, preprocessing, model/checkpoint identity, metrics, subgroup reporting, failure handling and one-time execution gate before any inference.

## Step 15 external-evaluation contract freeze — 5 August 2026

- Created and fully executed `step_15_frozen_external_evaluation_contract/step_15.ipynb` plus its descriptive-name notebook, durable generator, README and repair log.
- Result level: `EXTERNAL_EVALUATION_CONTRACT_FROZEN_AWAITING_AUTHORIZATION`; all 10/10 readiness requirements passed.
- Frozen the complete 20-case external cohort with zero exclusions: 15 tumour-positive patients, five tumour-negative controls and 24 source lesion-folder proxies.
- Applied only Step 01 train-derived burden edges. Patient coverage is Q1 = 1, Q2 = 7, Q3 = 0 and Q4 = 7; lesion-proxy coverage is Q1 = 0, Q2 = 2, Q3 = 4 and Q4 = 18. These distributions are reporting strata and cannot change preprocessing or model policy.
- External median slice count is 127 versus 263 in training. All compared geometry features remain within observed training ranges; one external patient is a train-IQR outlier for slice spacing.
- Reverified all three checkpoint hashes, the MobileNetV2UNet source hash, Step 03 frozen policy identity, the final Step 14 signed package and all 60 normalized NIfTI paths.
- Frozen external preprocessing reproduces the original broad HU window `[-160,240]`, uint8 rounding, 256 × 256 grid, predicted-liver ROI checkpoint/threshold/largest-3D component/padding, maximum control-recall probability fusion, global threshold 0.70 and no post-processing.
- Predeclared the complete metric suite, train-derived subgroups, 10,000-patient bootstrap, integrity gates, validation-era performance floors, failure action and one-time execution ledger rule.
- Repaired one safe execution error: Step 14 normalized paths are relative to the Step 14 phase directory, not the Part 2 root. The generator/notebook now resolve them correctly; all existing normalized files were preserved.
- Validation: both notebook names pass `nbformat` and AST parsing; the short notebook executed top-to-bottom with zero errors; all signed outputs exist. Current Step 15 signature: `aaca93be9d1d64e63807fb10f642b98cd0371504667f55ad69a14d8f1246f77d`, superseding the earlier time-dependent signature after the user's final rerun.
- Model loaded: false. Inference performed: false. Formal external evaluation performed: false. Local LiTS test images, masks, statistics and loaders accessed: false.
- Single next action: provide the exact authorization sentence recorded in the Step 15 gate, then create/run Step 16 once under the frozen contract. No threshold sweep, tuning, case exclusion or result-driven rerun is allowed.

## Step 16 creation and safe authorization preflight — 5 August 2026

- Created `step_16_one_time_external_evaluation_after_explicit_authorization/step_16.ipynb` plus the descriptive-name notebook, durable generator, README, repair log and isolated output/cache directories.
- The guarded notebook implements the complete authorized path: native-to-256 standardization, predicted-liver ROI generation, control/recall inference, maximum fusion at 0.70, probability caching, patient/slice/lesion/negative-control/subgroup metrics, bootstrap uncertainty, predeclared acceptance gate, evidence inventory, dashboard, failure cases, signature and one-time ledger sealing.
- Default result level: `EXTERNAL_EVALUATION_AUTHORIZATION_REQUIRED`; this is an expected successful safe-preflight state, not an execution failure.
- All 12/12 prerequisite checks passed, including the current Step 15 signed package, all three checkpoint hashes, 20 unique external patients, 15 positive and five negative cases, all 60 normalized files, authoritative manifest identity, frozen fusion/threshold/post-processing and local-test exclusion.
- The default Run All created no run ledger and no probability cache files. It did not import/load the model for inference and did not execute any expensive model cell.
- Validation: both notebook names pass `nbformat`; all six code cells parse; the short notebook executed top-to-bottom with zero errors; the signed preflight artifacts match.
- Model loaded: false. Inference performed: false. Formal external evaluation performed: false. Local LiTS test images, masks, statistics and loaders accessed: false.
- Single next action: the user must supply the exact authorization sentence frozen in Step 15. Then record that sentence, a UTC timestamp and a new UUID in both the generator and notebook before the one allowed expensive Run All.

## Step 16 authorized one-time external evaluation — 5 August 2026

- The user supplied the exact frozen authorization sentence. It was recorded at `2026-08-05T14:49:18.3333060Z` with one-time UUID `61d1f140-c8b4-436a-928a-1e4b6f7c0b56` in both the generator and notebook.
- Before inference, repaired ledger/signature ordering so the final sealed ledger is signed and never modified afterward; also made the authorization state accurately record completed model loading/inference. No cache existed when repaired.
- Executed the full frozen pipeline once over all 20 3D-IRCADb-01 cases and 2,823 slices. Created 20 finite, identity-checked patient probability caches and sealed the run ledger at `sealed_complete` with reruns prohibited.
- Result level: `EXTERNAL_EVALUATION_COMPLETE`; decision `REPORT_EXTERNAL_GENERALIZATION_PASS_NO_TUNING`; all 11/11 predeclared mandatory integrity, performance and reporting rows passed.
- Global Dice 0.847977; pixel precision 0.823699; pixel recall 0.873729. Mean positive-patient Dice 0.711196 (95% bootstrap CI 0.567803–0.827823), median 0.835514 and minimum 0.012992.
- Q1 positive-slice detection 50.00%; positive predicted-empty rate 9.86%; empty-slice false-positive rate 11.57%. All values independently recomputed exactly from saved patient/slice counts.
- Burden behavior is heterogeneous: Q1 patient Dice 0.717752; Q2 mean 0.544269 with minimum 0.012992; Q4 mean 0.877186. Smallest-lesion detection is 63.64%, rising to 100% in Q4.
- All five tumour-negative controls had at least one predicted pixel. Cases 7 and 14 dominate the count. Under the frozen accepted hepatic-tumour labels these are false positives; biological meaning is unresolved.
- Step 16 signature combined SHA-256: `4fb3d6d14010a982912c9ca8e85854e326960fa4a60543da47a378c6fd1b13aa`; all 18 signed hashes and all 32 inventory/cache hashes were independently verified.
- Local LiTS test accessed: false. No tuning, threshold sweep, post-processing change, case exclusion or result-driven rerun occurred.
- Single next action: validate and package the sealed evidence read-only; never rerun Step 16.

## Step 17 external evidence validation and data card — 5 August 2026

- Created and fully executed `step_17_external_evaluation_evidence_validation_and_data_card/step_17.ipynb` plus its descriptive notebook, generator, README and repair log.
- Result level: `EXTERNAL_GENERALIZATION_EVIDENCE_VALIDATED_WITH_CAVEATS`; Step 16's predeclared external pass is confirmed, not changed.
- Independently verified the sealed ledger, run UUID, all 18 signed Step 16 artifacts, all 32 evidence/cache inventory rows and all 20 probability caches. Recomputed all headline metrics and 11 gate decisions exactly.
- Quantified the main caveats: worst positive patient `ircadb_18` Dice 0.012992; smallest-lesion detection 14/22 (63.64%); all five negative controls had predictions totaling 52.936 ml on the standardized physical grid.
- Produced a signed external-evaluation data card, patient-risk profile, negative-control physical-volume audit, lesion difficulty validation, probability histograms, background-dominated descriptive reliability, evidence dashboard and three-case failure atlas.
- Visual review shows the largest negative-control predictions overlap visible low-attenuation structures in cases 7 and 14, but biological meaning cannot be inferred without expert/source-annotation review. They remain false positives under the frozen truth.
- Prohibited claims are explicit: no clinical readiness, universal generalization, state-of-the-art or zero-false-positive claim.
- Step 17 signature: `4b0f2b13eef3f5a70a41eca94b826703f3daeac85b17b354886c3357ac4c3b48`.
- Validation: both notebook names pass `nbformat`; all code cells parse and execute with zero errors; dashboard and failure atlas visually inspected; all signed hashes verified.
- Inference performed in Step 17: false. Step 16 rerun: false. Tuning performed: false. Local LiTS test accessed: false.
- Single next action: optionally obtain expert/source-annotation review for the preserved failure cases or begin a separately frozen second-external-dataset contract. Step 16 is complete and immutable.

## Step 18 source-label concordance and failure adjudication — 5 August 2026

- Created and fully executed `step_18_source_label_concordance_and_failure_adjudication/step_18.ipynb` plus its descriptive-name notebook, durable generator, README and repair log.
- Result level: `SOURCE_LABEL_CONCORDANCE_COMPLETE_EXPERT_REVIEW_REQUIRED`; all six requirements passed. Step 16's sealed pass and all reported metrics remain unchanged.
- Reconstructed and aligned 72 original source DICOM mask folders across the five tumour-negative controls and worst positive case. All masks had zero missing CT instances, matching frozen-grid shapes and position errors within tolerance.
- Case 7: 5,334/6,174 predicted pixels (86.39%) overlap the excluded generic `tumor` mask. Case 14: 6,694/7,527 (88.93%) overlap `metastasectomie`, with source-label Dice 0.851925.
- Case 18 remains a genuine accepted-truth failure under the frozen contract: only 16 predicted pixels intersect `livertumor`, recall 1.41%, Dice 0.012992. Case 20 has 71.66% prediction overlap with the source gallbladder mask.
- Case 5 does not overlap either excluded adrenal/surrenal tumour mask; case 11 contains only one predicted pixel. Only cases 7 and 14 exceeded 5% overlap with an excluded annotation.
- Source-label overlap establishes annotation concordance, not biological truth. No retrospective relabeling, metric revision, model tuning or clinical claim is justified without blinded expert/source review.
- Repaired the stale Jupyter launcher fallback, removed an optional `tabulate` dependency, corrected the required DICOM `(y,x)` to frozen NIfTI `(x,y)` transpose, and made panels select specific annotations rather than the broad `skin` container. All repairs are durable in the generator and logged.
- Generated full concordance/alignment/adjudication CSVs, labelled panels, overlap dashboard, report, provenance, gate and signed evidence package. Combined Step 18 signature: `1f1f222d300e689a5e36787b4b7da60b0b27ddb594bab8d8aad736fe3ed35b6f`.
- Inference performed: false. Step 16 rerun: false. Tuning performed: false. Local LiTS test accessed: false.
- Single next action: optional blinded medical/source-annotation review of the preserved Step 18 panels. If such expertise is unavailable, the scientifically appropriate action is to stop result-driven analysis or separately freeze a second independent external-dataset contract—not to alter Step 16.

## Step 19 external-evidence manuscript integration — 5 August 2026

- Created and fully executed `step_19_external_evidence_manuscript_integration/step_19.ipynb` plus its descriptive-name notebook, durable generator, README and repair log.
- Result level: `MANUSCRIPT_EXTERNAL_EVIDENCE_UPDATED_OWNER_INPUT_REQUIRED`; all six requirements passed. The external pass is integrated, but the failed LiTS minimum-patient guardrail remains unchanged.
- Verified the relevant Step 5 evidence inventory and all signed Step 7, 17 and 18 input artifacts before manuscript synthesis. Recomputed positive-patient means from sealed patient tables and reconciled them to the reported summaries.
- LiTS evidence: global Dice 0.767696, mean positive-patient Dice 0.507297 and minimum 0. External evidence: global Dice 0.847977, mean 0.711196 (95% bootstrap CI 0.567803–0.827823), median 0.835514 and minimum 0.012992.
- Cross-cohort differences are explicitly descriptive only because acquisition, tumour prevalence, denominators and label semantics differ. The claim that the external cohort “outperformed” LiTS is prohibited as an inferential claim.
- Preserved shared failure evidence: LiTS smallest-lesion detection 60.3%; external 63.64%; external five-of-five negative controls had predictions totaling 52.936 ml; cases 7 and 14 retain annotation-semantic caveats.
- Produced a revised manuscript draft with external validation, technical report, updated limitations and claim boundaries, cohort/performance/patient/failure tables, claim matrix, chart map, and two paper-ready figures. Both figures were visually inspected after repairing legend overlap and duplicate outlier rendering.
- Submission remains closed because Step 11 owner fields are incomplete and the exact scholarly 3D-IRCADb-01 citation has not been verified. No manuscript submission or payment was authorized or performed.
- Combined Step 19 signature: `d6075d9fc6d5d802316f2e3e7d9f7ee3f379ae9fc6919053e292fb496b3e0fb3`.
- Inference performed: false. Test source files reopened: false. Test/external rerun: false. Tuning: false. Submission action: false.
- Single next action: complete the owner declarations and verify the exact 3D-IRCADb-01 scholarly citation if submission remains desired. Expert source-annotation review is optional but required before any biological reinterpretation.

## Step 20 external citation verification and owner readiness — 6 August 2026

- Created and fully executed `step_20_external_citation_verification_and_owner_readiness/step_20.ipynb` plus its descriptive-name notebook, durable generator, README and repair log.
- Result level: `EXTERNAL_CITATION_VERIFIED_OWNER_DECLARATIONS_REQUIRED`; all six requirements passed. The Step 19 manuscript evidence package and all 17 signed artifacts were independently verified.
- Verified the exact 3D-IRCADb-01 citation from the current official IRCAD page: Soler L, Hostettler A, Agnus V, Charnoz A, Fasquel J, Moreau J, Osswald A, Bouhadjar M, Marescaux J. “3D image reconstruction for comparison of algorithm database: A patient specific anatomical and medical image database.” IRCAD, Strasbourg, France, Technical Report (2010).
- The official page confirms 20 patients (10 women, 10 men), hepatic tumours in 75% of cases, and CC BY-NC-ND 4.0 International licensing. It provides no DOI, so no DOI was fabricated.
- Created a verified `Soler2010IRCADb` BibTeX addendum, citation-patched manuscript copy, official-source snapshot/evidence table, manuscript patch plan, advisory dataset-terms/data-availability statements and complete owner blocker matrix.
- Owner-controlled submission state remains 0/14 fields complete. Dataset/licence language is only an advisory draft until the owner verifies intended sharing, institutional policy and venue requirements.
- Combined Step 20 signature: `316c298dcf6e4f027de9d039ce7c2aa8637bc6e37971613bb1b46819008a751d`.
- Inference: false. Dataset download: false. Test images/source reopening: false. Tuning: false. Submission/payment action: false.
- Single next action: the owner must complete and certify the 14 Step 11 fields using the verified citation and advisory dataset statements. No further scientific notebook is required until owner inputs or a new research question are supplied.

## Step 21 terminal evidence chain and project handoff — 6 August 2026

- Created and fully executed `step_21_terminal_evidence_chain_and_project_handoff/step_21.ipynb` plus its descriptive-name notebook, durable generator, README and repair log.
- Result level: `PROJECT_EVIDENCE_CHAIN_VERIFIED_OWNER_ACTIONS_PENDING`; all eight terminal requirements passed and all 20 prior phase gate files were inventoried.
- Inventoried 19 signature JSON files and independently audited 192 signed-artifact mappings. Every current mapping matches its SHA-256.
- The audit correctly found two Step 16 preflight mismatches for `authorization_state.json` and `gate_result.json`. These are expected historical differences: the preflight signature captured the authorization-required state, then the authorized one-time run intentionally superseded both files and produced the authoritative final Step 16 signature. Step 21 preserves them as `superseded_preflight`, not as current-integrity failures.
- Terminal scientific state: authoritative corrected manifest preserved; LiTS held-out evaluation complete with failed minimum-patient formal gate; separate 3D-IRCADb-01 contract passed with patient/small-lesion/negative-control caveats; external citation and licence verified; manuscript evidence complete but not submission-ready.
- Produced the complete phase gate chain, recursive artifact index, detailed signature verification, signature-file inventory, terminal scientific summary, unresolved-action register, chart map, visually inspected 20-phase evidence-chain figure, terminal project data card and terminal handoff.
- Combined Step 21 signature: `980669a07052035505b4239fe6a7c5e78958c9eb6fbf15381add78f924688282`.
- Inference: false. Test images/source reopening: false. Test/external rerun: false. Tuning/download/submission/payment: false.
- Terminal next action: the project owner completes and certifies the 14 Step 11 fields if submission remains desired. Otherwise stop and preserve the archive. Any future scientific phase requires a genuinely new predeclared question and new evidence.

## LiTS dataset provenance consolidation — 6 August 2026

- Created `08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md` as the canonical reusable account of the LiTS acquisition history.
- Recorded the owner-supplied source mapping: Kaggle Part 1 for volumes 0-50; Kaggle Part 2 for 51-69 and 100; Hugging Face CADS `0004_lits` for 70-99 and 101-130.
- Reconciled that mapping with the retained staging `dataset_version.json`, partial Kaggle `download_log.json`, historical pipeline notebook, legacy PNG source registry, corrected-build records and current provider pages.
- Verified the current local filename inventory contains 131 CT volumes and 131 segmentations with unique IDs 0-130 and no missing IDs. Reverified the authoritative manifest SHA-256 as `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`.
- Documented that the project-internal test split is a holdout from the 131 annotated LiTS training volumes, not the official 70-volume challenge test set.
- Preserved the licence uncertainty rather than inventing one mixed-source licence: the current Kaggle page displays CC BY-NC-ND 4.0, while the CADS LiTS README displays CC BY-NC-SA 4.0. Raw-data redistribution remains discouraged pending source-specific terms review.
- No dataset file, corrected build, manifest, split, model, cache or signed phase output was modified. Locked test images, masks and statistics were not opened or recomputed.
- Publication fields remain inactive by owner choice. Future work may cite the new provenance record and existing Step 01 EDA without repeating the acquisition audit.


## Step 00 model improvement — Phase 0 & Phase 1 (7 August 2026)

New development-only research programme under
`step_00_model_improvement\` (phases 0-5) to try to improve tumour
segmentation on the local corrected LiTS build. This is independent of the
completed Mark 4E baseline and cannot reopen that evidence.

### Phase 0 — contract & holdout seal (COMPLETE, executed)
- Created `step_0_contract` generator + notebook; executed end to end.
- Verified manifest SHA-256 matches `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`; test loader lock asserted (raises).
- Sealed 8-volume one-time internal holdout: **EVAL_VOLUMES = [4,25,44,64,83,84,90,100]** (low-burden 83/25; large-lesion 4/100; V104 analogs 64/90; V116 analogs 84/44). 96 dev train volumes remain.
- Wrote devices: `EVAL_VOLUMES.json`, `data_card.json`, `sampling.json`, `gate_result.json` (status STEP_0_CONTRACT_AND_HOLDOUT_SEALED, decision PROCEED).
- `test_images_accessed: false`, `external_dataset_accessed: false`.

### Phase 1 — smoke gate (generator built + preflight, NOT executed)
- Built `step_1_smoke` generator -> `step_1_smoke.ipynb` (16 cells, all parse).
- Protocol: each arm (Control, C1 high-res ROI, C2 analog sampler, C3 capped recall) cold-start overfits a deterministic 16-slice dev set; gate hard micro-Dice >= 0.80. GPU only.
- Preflight OK: dev split (35,836 rows / 96 vols), overfit selection (8 pos + 8 neg), C1 ROI crop to 256x256, model forward, both loss forwards.
- Heavy GPU training intentionally left for manual execution.

### Next action
Run `step_1_smoke.ipynb` end to end on the GPU venv; then proceed to Step 2 full training for passing arms.

Status governing model-improvement pipeline: Phase 0 sealed; Phase 1 generated/validated awaiting manual GPU run. Project-wide baseline remains `FINAL_PROJECT_COMPLETE` at Mark 4E.

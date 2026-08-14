# Dataset and Notebook Evidence Extraction

Last updated: 2026-08-08

Scope: extracted from `Practice/`, `mark 1/`, and `mark 1 (part 2)/` under `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`.

## 1. Current High-Level Status

The project has a corrected, auditable LiTS dataset and a completed model evidence chain. The final status is mixed:

- Dataset integrity and train/validation characterization passed.
- Validation freeze passed after pixelwise fusion.
- One-time LiTS held-out evaluation was completed, but formal model acceptance failed because one positive test patient had effectively zero Dice.
- A separate 3D-IRCADb-01 external evaluation passed under its frozen contract, with caveats.
- No further test rerun or result-driven threshold/model tuning is allowed on sealed LiTS evidence.

## 2. Authoritative Dataset Identity

- Project root: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`
- Authoritative evidence store: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\Practice`
- Historical model workspace: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1`
- Phase-isolated continuation workspace: `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver\mark 1 (part 2)`
- Corrected dataset build: `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- Authoritative manifest hash: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- Canonical sample key: `(volume_id, slice_index)`
- Sample id style: `v{volume_id:03d}_s{slice_index:04d}`
- Stored model image/mask size: 256 x 256
- Volumes: 131 total, volume IDs 0 through 130
- Total slices: 58,638
- Tumor-positive slices: 7,114
- Empty/negative tumor slices: 51,524
- Total positive tumor pixels at 256 x 256 mask resolution: 4,489,160

## 3. Locked Split Design

The project uses patient/volume-disjoint splits:

| Split | Volume IDs | Volumes | Slices | Tumor-positive slices | Tumor pixels | Positive-slice rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | 0-103 | 104 | 40,667 | 4,930 | 2,589,463 | 0.1212285 |
| Validation | 104-116 | 13 | 10,685 | 1,042 | 659,487 | 0.0975199 |
| Test | 117-130 | 14 | 7,286 | 1,197 | 1,366,941 | 0.1642877 |

Rules:

- Split by volume, never by slice.
- Training reads train only.
- Threshold/model selection reads validation only.
- Test is sealed and is not available for further tuning or reruns.
- Dataset-level integrity checks may confirm existence, geometry, labels, and provenance, but test performance must not drive development.

## 4. Source Reconstruction and Data Quality Findings from Practice

The original `Practice/dataset summary.txt` recorded that training was blocked until authoritative source reconstruction and validation passed. Important findings:

- `LiTS_masks` had 58,638 mask files, 131 volumes, complete continuous keys, 256 x 256 dimensions, RGB channels with no disagreement, binary 0/1 values, and no unreadable files.
- `lits-png\dataset_6\dataset_6` had 58,638 candidate 256 x 256 images with complete keys.
- Only 15,817 embedded lesion masks and 15,868 embedded liver masks were locally available in that partial PNG source package.
- All 15,817 locally available embedded lesion masks were byte-identical to the corresponding renamed masks in `LiTS_masks`.
- Across those 15,817 verified triples, tumor pixels outside liver mask were 0.
- `Liver Img Dataset` had 58,638 512 x 512 grayscale images with matching numerical keys, but it was not spatially proven compatible with the masks and was marked legacy/classification-only.
- `lits_df.csv` had 58,638 rows and unique filepath-derived keys, but `instance_number` wrapped after 255 and produced 30,105 duplicate `(study_number, instance_number)` keys.
- The upstream `tumor_mask_empty` field was semantically inverted/misleading and had to be treated as tumor-present after correction.
- Upstream `lits_train.csv`, `lits_test.csv`, and `lits_probe.csv` were not accepted as canonical research splits because they covered only 70.88% of the dataset and duplicated probe/test.

Earlier `Practice/unified_dataset_preparation_outputs/20260724_211625/dataset_readiness.json` showed an intermediate state:

- Source key count: 58,638
- Source volume count: 131
- NIfTI complete pairs: 131
- NIfTI shape matches: 131
- NIfTI affine matches: 79
- Automatic build gates pass: false
- Training ready: false
- Required action then: complete orientation review/correction before training

This was later superseded by corrected build and loader/overfit gate evidence.

## 5. Corrected Loader and Overfit Gate

From `Practice/verified_loader_overfit_outputs/gate_result.json`:

- Dataset build: `build_corrected_20260713_214847_v2`
- Manifest SHA-256: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- Loader gate passed: true
- Overfit gate passed: true
- All gates passed: true
- Best hard micro Dice: 0.8163059889383042
- Required hard micro Dice: 0.8
- Selected training slices: 16
- Test images accessed: false
- Decision: proceed to manifest integration and staged baseline training

Loader pixel parity samples confirmed loaded tumor pixel counts matched manifest tumor pixel counts.

## 6. Mark 1 Validation and Model Evidence

### Mark 4E Fusion Validation

From `mark 1/mark_4e_outputs/mark_4e_gate_result.json`:

- Status: `mark_4e_fusion_pass`
- Selected policy: `maximum`
- Fusion equation: `p_fused(x)=max(p_control(x), p_recall(x))`
- Selected threshold: 0.70
- Metric definition: mean Dice over nine tumor-positive validation patients
- Test images accessed: false
- Decision: `FREEZE_FUSION_POLICY_AND_THRESHOLD`

Selected validation metrics:

| Metric | Value |
|---|---:|
| Mean positive-patient Dice | 0.3770866927 |
| Volume 104 Dice | 0.1166269753 |
| Volume 116 Dice | 0.0104735533 |
| Q1 detected percent | 50.5703422053 |
| Positive predicted-empty percent | 27.4472168906 |
| Empty-slice false-positive percent | 5.5480659546 |

This passed six temporary validation targets, but it was not final test success.

## 7. Part 2 Step Evidence

### Step 01: Pretraining Dataset Characterization

From `mark 1 (part 2)/step_01_pretraining_dataset_characterization/outputs/pretraining_dataset_gate.json`:

- Status: `pretraining_dataset_audit_complete`
- Result level: `DIAGNOSTIC_COMPLETE`
- Decision: `PASS_FREEZE_DATA_CARD_AND_PROCEED_TO_FUSION_CONFIRMATION`
- Train/validation rows: 51,352
- Train/validation volumes: 117
- Critical integrity failures: 0
- Critical geometry failures: 0
- Critical label failures: 0
- Critical ROI failures: 0
- Test images accessed: false
- Selected sampling policy: `uniform_patient_aware_existing_sampler_no_change`
- All 12 mandatory targets passed, including manifest identity, train/validation leakage, morphology profile, HU profile, label containment, ROI coverage, focus V104/V116 presence, and train-derived bins.

### Step 02 and Step 03

The Step 21 phase chain reports:

- Step 02 `fusion_freeze_confirmation`: `VALIDATION_FREEZE_PASS`
- Step 03 `final_inference_policy_freeze`: `VALIDATION_FREEZE_PASS`

These froze the final inference policy before test access.

### Step 04: One-Time Locked LiTS Test Evaluation

From `mark 1 (part 2)/step_04_one_time_locked_test_evaluation_after_explicit_authorization/outputs/gate_result.json`:

- Result level: `FINAL_TEST_COMPLETE`
- Status: `one_time_locked_test_evaluation_complete`
- All mandatory targets passed: false
- Decision: `REPORT_FINAL_TEST_GATE_FAILURE_NO_TUNING_NO_RERUN`
- Test images accessed: true
- One-time run id: `871d289b-bf6b-4346-978f-2df02ade26ab`
- Selected configuration:
  - Fusion: maximum
  - Threshold: 0.70
  - Post-processing: none

Held-out LiTS metrics:

| Metric | Value |
|---|---:|
| Global Dice | 0.7676962585 |
| Global pixel precision | 0.8443941209 |
| Global pixel recall | 0.7037714137 |
| Mean positive-patient Dice | 0.5072968406 |
| Median positive-patient Dice | 0.5651373760 |
| Minimum positive-patient Dice | 0.000000000761 |
| Q1 positive-slice detection percent | 47.5524475524 |
| Positive predicted-empty percent | 9.9415204678 |
| Empty-slice false-positive percent | 3.8101494498 |
| Sample coverage fraction | 1.0 |
| Unique sample id fraction | 1.0 |
| Finite probability fraction | 1.0 |

Acceptance target failure:

- Required minimum positive-patient Dice: 0.01
- Observed minimum positive-patient Dice: effectively 0
- Therefore formal LiTS model acceptance failed, despite strong aggregate Dice.

### Step 16: One-Time External 3D-IRCADb-01 Evaluation

From `mark 1 (part 2)/step_16_one_time_external_evaluation_after_explicit_authorization/outputs/gate_result.json`:

- Result level: `EXTERNAL_EVALUATION_COMPLETE`
- Decision: `REPORT_EXTERNAL_GENERALIZATION_PASS_NO_TUNING`
- Formal external evaluation performed: true
- Local LiTS test accessed: false
- Rerun permitted: false
- Cohort: 20 patients total, 15 positive, 5 negative, 0 excluded

External metrics:

| Metric | Value |
|---|---:|
| Global Dice | 0.8479770814 |
| Global pixel precision | 0.8236994368 |
| Global pixel recall | 0.8737293019 |
| Mean positive-patient Dice | 0.7111958843 |
| Median positive-patient Dice | 0.8355140187 |
| Minimum positive-patient Dice | 0.0129922862 |
| Q1 positive-slice detection percent | 50.0 |
| Positive predicted-empty percent | 9.8591549296 |
| Empty-slice false-positive percent | 11.5742793792 |
| Sample coverage fraction | 1.0 |
| Finite probability fraction | 1.0 |

### Step 21: Terminal Evidence Chain

From `step_21_terminal_evidence_chain_and_project_handoff`:

- Part 2 phases inventoried: 20/20
- Current signed-artifact mappings verified
- LiTS held-out result complete, but formal model acceptance failed because minimum positive-patient Dice was 0
- 3D-IRCADb-01 separate frozen external contract passed with preserved caveats
- External citation/license verified: `Soler2010IRCADb`, CC BY-NC-ND 4.0
- Submission readiness: false
- Owner fields complete: 0/14

Terminal scientific summary:

| Evidence area | Status | Key result | Claim boundary |
|---|---|---|---|
| Authoritative corrected dataset | verified | manifest SHA-256 `575a6fc...889` | patient-disjoint build; historical source read-only |
| LiTS held-out evaluation | complete formal acceptance failed | global Dice 0.767696; minimum patient Dice 0.000000 | no further test reuse or test-driven tuning |
| 3D-IRCADb-01 external evaluation | separate frozen contract passed | global Dice 0.847977; mean positive-patient Dice 0.711196 | single public cohort with patient/lesion/negative-control caveats |
| External citation/license | verified | Soler2010IRCADb; CC BY-NC-ND 4.0 | no redistribution authorization; owner reviews language |
| Manuscript package | complete with owner blockers | citation-patched manuscript and signed evidence package exist | not submission-ready; owner gate closed |

## 8. Folder Roles

### Practice

Canonical evidence and provenance store. Contains source reconstruction notes, dataset pairing/orientation forensics, corrected loader evidence, overfit gate evidence, and historical exploratory/ablation outputs. Treat as read-only evidence unless a new task explicitly authorizes changes.

### mark 1

Historical modeling and validation workspace. Key contribution is the Mark 4 series, especially Mark 4E fusion validation that froze maximum fusion at threshold 0.70 after validation targets passed.

### mark 1 (part 2)

Phase-isolated continuation workspace. Contains the structured evidence chain from dataset characterization through validation freeze, one-time LiTS test, final package, external dataset work, citation verification, and terminal handoff.

## 9. What Is Safe to Claim

Supported:

- The corrected LiTS dataset has an auditable manifest and volume-wise split.
- Train/validation dataset audit passed with zero critical integrity, geometry, label, and ROI failures.
- The loader and 16-slice overfit gate passed before full modeling.
- Validation fusion policy was frozen before test access.
- One-time held-out LiTS evaluation produced strong aggregate metrics.
- Formal LiTS acceptance failed due to a catastrophic minimum-patient Dice failure.
- Separate external 3D-IRCADb-01 evaluation passed under a frozen contract, with caveats.

Not supported:

- Clinical readiness.
- Universal generalization.
- Final LiTS model acceptance.
- Any further LiTS test tuning, rerun, threshold adjustment, or post-hoc rescue.
- Submission readiness until owner declarations and venue-specific fields are completed.

## 10. Immediate Next Research Boundary

Do not reopen sealed LiTS test evaluation. If scientific work resumes, it should be under a new predeclared research question using train-only development folds, not legacy validation/test outcomes. The remembered later direction is an independent research reset: derive patient-disjoint train-only folds from the 104 training patients, freeze metrics/compute/baseline decisions, and only then explore a lightweight refinement direction without touching sealed cohorts.


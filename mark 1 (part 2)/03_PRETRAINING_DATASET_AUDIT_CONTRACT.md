# Pre-Training Dataset Audit Contract

## Purpose

The immediate next phase is a dataset-characterization notebook, not training. It must describe the train and validation populations deeply enough to justify any later sampling, normalization, augmentation or architecture decision while avoiding test leakage.

## Required notebook and output locations

- Phase folder: `mark 1 (part 2)\step_01_pretraining_dataset_characterization\`
- Notebook: `step_01_pretraining_dataset_characterization\step_01_pretraining_dataset_characterization.ipynb`
- Outputs: `step_01_pretraining_dataset_characterization\outputs\`
- Phase notes: `step_01_pretraining_dataset_characterization\README.md`
- Error record when needed: `step_01_pretraining_dataset_characterization\error_fixes.md`

The notebook must read source NIfTI files for physical geometry and HU analysis. Derived PNGs may be used only where their existing preprocessing is the intended object of analysis.

## Scope

- Allowed: train and validation images, masks, metadata and previously saved validation artifacts.
- Training-derived bins/policies: train only.
- Allowed validation use: distribution comparison, coverage checks and phenotype/outlier reporting.
- Forbidden: test loader, test images, test masks, test EDA, test thresholds or test-informed decisions.

## Section A — manifest and integrity profile

Required checks:

- manifest hash, row/column counts and schema;
- required-column missingness;
- unique `sample_id` and unique `(volume_id, slice_index)`;
- path existence and source-hash coverage;
- split volume disjointness;
- allowed `verification_status`, spatial status and transform values;
- manifest pixel counts versus sampled loaded masks;
- duplicate and near-duplicate source-hash checks across train and validation;
- exclusions and unexpected sentinel values.

Required outputs:

- `manifest_quality_summary.csv`;
- `integrity_failures.csv` even if empty;
- `split_summary.csv`;
- `data_quality_gate.json`.

Critical stop conditions:

- duplicate patient/source identity across train and validation;
- broken manifest grain or required paths;
- unresolved affine/label alignment failure;
- non-binary or mismatched target masks;
- manifest hash mismatch.

## Section B — acquisition and physical geometry

Per-volume measurements:

- NIfTI shape and number of slices;
- voxel spacing and slice thickness;
- affine determinant, axis codes and orientation;
- physical field of view;
- image/segmentation shape and affine agreement after approved transform;
- liver and tumour physical volume in millilitres;
- frozen ROI dimensions, physical dimensions, area ratio and tumour coverage.

Required summaries:

- train-versus-validation quantiles and standardized differences;
- outlier flags using train-derived robust bounds such as Q1/Q3 and IQR;
- explicit V104 and V116 rows;
- orientation/spacing/ROI dashboards.

Required outputs:

- `volume_geometry_profile.csv`;
- `geometry_split_comparison.csv`;
- `geometry_outliers.csv`;
- `geometry_dashboard.png`.

## Section C — tumour morphology and burden

Use 3D connected components on source segmentation after approved orientation handling.

Per-patient and per-lesion features:

- number of lesions;
- lesion voxel count and physical volume;
- equivalent spherical diameter;
- bounding-box dimensions and axial span;
- surface-to-volume or compactness proxy;
- tumour-to-liver volume ratio;
- tumour-positive slice count and continuity;
- multifocal versus solitary disease;
- central versus peripheral liver-position proxy when supportable.

Sampling strata must be derived using train data only. Record fixed bin edges and then map validation patients into those bins.

Required outputs:

- `patient_burden_profile.csv`;
- `lesion_component_profile.csv`;
- `train_derived_lesion_bins.json`;
- `lesion_distribution_dashboard.png`;
- `validation_stratum_coverage.csv`.

## Section D — HU contrast and appearance domain

From source NIfTI, measure per positive slice/lesion:

- tumour HU median, IQR, P10 and P90;
- non-tumour liver HU median and IQR;
- tumour-minus-liver contrast;
- robust contrast-to-noise ratio;
- visibility under broad `[-160,240]` and liver `[0,200]` windows;
- liver-background and extra-liver distributions where masks permit;
- volume-level robust intensity statistics.

Compare train and validation distributions without fitting a validation-specific transform. Find training analogs for V104 and V116 using standardized features whose scaler and distance definition are fit on train only.

Required outputs:

- `hu_contrast_per_slice.csv`;
- `hu_contrast_per_volume.csv`;
- `appearance_feature_profile.csv`;
- `validation_training_analogs.csv`;
- `v104_v116_domain_dashboard.png`;
- `intensity_distribution_dashboard.png`.

## Section E — label and spatial QC

Required checks:

- tumour containment inside liver;
- tumour coverage by frozen predicted-liver ROI;
- source segmentation values and binary derivation;
- disconnected components and tiny isolated components;
- slice-to-slice centroid and area jumps;
- image/mask edge alignment samples;
- affine and approved transform consistency;
- extreme annotation shapes for manual review.

Required outputs:

- `label_qc_per_volume.csv`;
- `label_qc_review_cases.csv`;
- `label_alignment_review.png`;
- `roi_coverage_review.csv`.

## Section F — model-relevant difficulty profile

Join dataset features with existing train/validation diagnostics only after feature computation is complete. Analyse whether failure is associated with:

- lesion volume/diameter/axial span;
- tumour burden and multifocality;
- HU contrast and liver variability;
- spacing and slice thickness;
- ROI compression;
- acquisition/outlier features.

Do not claim causality from correlation. Use this section to choose a stable train-derived sampling policy, not a per-validation-patient rule.

Required outputs:

- `difficulty_feature_associations.csv`;
- `patient_difficulty_profile.csv`;
- `difficulty_dashboard.png`.

## Final data card and gate

The notebook must create:

- `DATASET_DATA_CARD.md`;
- `sampling_policy.json`;
- `pretraining_dataset_gate.json`;
- `expected_vs_actual.csv`.

The data card must state dataset identity, splits, preprocessing, geometry ranges, lesion distributions, contrast distributions, known outliers, limitations and leakage safeguards.

The sampling policy must contain only train-derived definitions, including exact bin edges, weights/caps, random seed and rationale. It may recommend no sampling change.

The gate should be one of:

- `PASS_FREEZE_DATA_CARD_AND_PROCEED_TO_FUSION_CONFIRMATION`;
- `HOLD_FIX_DATA_OR_LABEL_ISSUE`;
- `HOLD_DEFINE_ONE_BOUNDED_DATA_INTERVENTION`.

## Suggested parameters

- Random seed: `42`.
- 3D connectivity: report both 6-connectivity default and sensitivity to 26-connectivity for tiny components if material.
- Robust outlier rule: train Q1/Q3 with `1.5 x IQR`; label, do not delete automatically.
- Histogram bins: fixed HU bins across compared populations.
- Minimum saved table precision: six decimals for metrics, full integers for counts.
- No stochastic subsampling for per-volume geometry or lesion components.
- If slice sampling is necessary for expensive HU work, make it deterministic, stratified and record coverage.

## Completion definition

This phase is complete only when every required output exists, source and row counts reconcile, visualizations are readable, the test-lock assertion passes, and the data card and sampling policy are frozen or a concrete blocker is recorded.

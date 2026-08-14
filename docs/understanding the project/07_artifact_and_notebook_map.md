# Artifact and Notebook Map

All paths below are relative to `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\Liver`.

## Dataset and loader

| Purpose | Notebook/artifact |
|---|---|
| Initial EDA | `Practice/lits_eda.ipynb` |
| Dataset pipeline | `Practice/dataset_pipeline.ipynb` |
| Pairing forensics | `Practice/dataset_pairing_forensics.ipynb` |
| Dataset validation/promotion | `Practice/dataset_validation_and_promotion.ipynb` |
| Corrected preparation | `Practice/unified_dataset_preparation_and_eda.ipynb` |
| Forensic summary | `Practice/dataset_pairing_forensics_results/forensic_summary.json` |
| Verified loader and overfit | `Practice/verified_manifest_loader_and_overfit_gate.ipynb` |
| Overfit gate | `Practice/verified_loader_overfit_outputs/gate_result.json` |
| Overfit status | `Practice/verified_loader_overfit_outputs/overfit_status.json` |

## Baseline and diagnostics

| Stage | Notebook | Primary output directory |
|---|---|---|
| 5-epoch smoke | `Practice/manifest_baseline_5epoch_smoke_test.ipynb` | `Practice/manifest_baseline_smoke_outputs` |
| Validation geometry | `Practice/validation_volume_mask_orientation_forensics.ipynb` | `Practice/validation_volume_forensics_outputs` |
| Patient-aware baseline | `Practice/baseline_25epoch_patient_aware_training.ipynb` | `Practice/patient_aware_baseline_outputs` |
| Epoch-10 continuation | `Practice/auto_cool_continue_patient_aware_to_epoch10.ipynb` | same baseline directory |

## Ablations

| Experiment | Notebook | Output directory |
|---|---|---|
| Lesion-balanced sampler | `Practice/patient_lesion_balanced_sampler_ablation.ipynb` | `Practice/patient_lesion_balanced_outputs` |
| Recall-aware loss | `Practice/recall_aware_focal_tversky_loss_ablation.ipynb` | `Practice/recall_aware_loss_outputs` |
| Composite loss | `Practice/stabilized_composite_loss_ablation.ipynb` | `Practice/stabilized_composite_loss_outputs` |
| Appearance forensics | `Practice/appearance_domain_robustness_forensics.ipynb` | `Practice/appearance_domain_forensics_outputs` |
| Intensity robustness | `Practice/organ_normalized_intensity_robustness_ablation.ipynb` | `Practice/intensity_robustness_outputs` |
| Adjacent-slice 2.5D | `Practice/adjacent_slice_2_5d_context_ablation.ipynb` | `Practice/context_2_5d_outputs` |
| 3D post-processing | `Practice/validation_3d_postprocessing_ablation.ipynb` | `Practice/validation_3d_postprocessing_outputs` |
| Multi-task localization | `Practice/multitask_liver_tumor_localization.ipynb` | `Practice/multitask_liver_tumor_outputs` |
| Multi-task continuation | `Practice/multitask_liver_tumor_epoch10_continuation.ipynb` | same multi-task directory |

## Most important current artifacts

Dataset identity:

- `Practice/verified_loader_overfit_outputs/gate_result.json`
- Corrected manifest at the dataset build path.

Strongest diagnostic checkpoint:

- `Practice/intensity_robustness_outputs/patient_aware_best.pth`
- Caveat: organ-assisted normalization.

Current deployable-pipeline experiment:

- `Practice/multitask_liver_tumor_outputs/multitask_best.pth`
- Best epoch: 8.

Current decision files:

- `Practice/multitask_liver_tumor_outputs/multitask_localization_gate_result.json`
- `Practice/multitask_liver_tumor_outputs/multitask_continuation_status.json`
- `Practice/multitask_liver_tumor_outputs/expected_vs_actual_results.csv`

Current detailed metrics:

- `Practice/multitask_liver_tumor_outputs/multitask_history.csv`
- `Practice/multitask_liver_tumor_outputs/best_validation_patient_metrics.csv`
- `Practice/multitask_liver_tumor_outputs/best_validation_per_slice.csv`
- `Practice/multitask_liver_tumor_outputs/best_validation_size_quartiles.csv`

Current figures:

- `Practice/multitask_liver_tumor_outputs/multitask_epoch_progress.png`
- `Practice/multitask_liver_tumor_outputs/multitask_results_dashboard.png`
- `Practice/multitask_liver_tumor_outputs/expected_vs_generated_predictions.png`
- `Practice/multitask_liver_tumor_outputs/multitask_target_audit.png`

Thermal record:

- `Practice/multitask_liver_tumor_outputs/multitask_thermal_log.csv`

## Generator scripts

The corresponding `Practice/create_*.py` scripts regenerate the newer notebooks. Important generators include:

- `create_verified_loader_overfit_notebook.py`
- `create_manifest_baseline_smoke_notebook.py`
- `create_patient_aware_baseline_notebook.py`
- `create_patient_lesion_balanced_ablation.py`
- `create_recall_aware_loss_ablation.py`
- `create_stabilized_composite_loss_ablation.py`
- `create_appearance_domain_robustness_forensics.py`
- `create_intensity_robustness_ablation.py`
- `create_2_5d_context_ablation.py`
- `create_validation_3d_postprocessing_ablation.py`
- `create_multitask_liver_tumor_localization.py`
- `create_multitask_epoch10_continuation.py`

## Historical warning

Do not use `Practice/next_phase_alignment_and_overfit.ipynb` for current modeling. It references the older invalid pairing generation.


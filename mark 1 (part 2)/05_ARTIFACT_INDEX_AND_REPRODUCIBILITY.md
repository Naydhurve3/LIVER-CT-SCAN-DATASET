# Artifact Index and Reproducibility

All paths below are relative to the project root unless absolute.

## Dataset

- Build: `D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2`
- Manifest: `manifests\slice_manifest.csv`
- Manifest SHA-256: `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`
- Raw authoritative NIfTI references are stored in manifest `source_volume_path` and `source_segmentation_path`.
- Download-source and licence record: `mark 1 (part 2)\08_LITS17_DATASET_PROVENANCE_AND_ACQUISITION_RECORD.md`.

## Historical authoritative implementation

- `Practice\` contains the corrected loader, model/loss framework, earlier notebooks and historical checkpoints.
- `src\framework\data\manifest_dataset.py`
- `src\framework\models\mobilenetv2_unet.py`
- `src\framework\losses\focal_dice.py`

## Mark 1 notebook sequence

| Phase | Notebook | Output directory |
|---|---|---|
| Mark 1 | `mark 1\mark_1_probability_contrast_localization_diagnostic.ipynb` | `mark 1\mark_1_outputs\` |
| Mark 2 | `mark 1\mark_2_roi_multiwindow_feasibility.ipynb` | `mark 1\mark_2_outputs\` |
| Mark 3 | `mark 1\mark_3_two_stage_multiwindow_overfit.ipynb` | `mark 1\mark_3_outputs\` |
| Mark 4 | `mark 1\mark_4_two_stage_validation_smoke.ipynb` | `mark 1\mark_4_outputs\` |
| Mark 4B | `mark 1\mark_4b_roi_probability_diagnostics.ipynb` | `mark 1\mark_4b_outputs\` |
| Mark 4C | `mark 1\mark_4c_two_channel_recall_ablation.ipynb` | `mark 1\mark_4c_outputs\` |
| Mark 4D | `mark 1\mark_4d_metric_reconciliation_v116_diagnostic.ipynb` | `mark 1\mark_4d_outputs\` |
| Mark 4E | `mark 1\mark_4e_checkpoint_fusion_validation.ipynb` | `mark 1\mark_4e_outputs\` |

## Controlling result artifacts

- Latest gate: `mark 1\mark_4e_outputs\mark_4e_gate_result.json`.
- Latest provenance: `mark 1\mark_4e_outputs\mark_4e_provenance.json`.
- Selected gate table: `mark 1\mark_4e_outputs\selected_gate_table.csv`.
- Fusion sweep: `mark 1\mark_4e_outputs\fusion_threshold_results.csv`.
- Patient metrics: `mark 1\mark_4e_outputs\fusion_patient_metrics.csv`.
- Fusion dashboard: `mark 1\mark_4e_outputs\fusion_validation_dashboard.png`.
- Patient heatmap: `mark 1\mark_4e_outputs\selected_fusion_patient_heatmap.png`.
- V116 panel: `mark 1\mark_4e_outputs\selected_fusion_v116_localization.png`.

## Probability caches and checkpoints

- Mark 4D cache root: `mark 1\mark_4d_outputs\probability_cache\`.
- Control cache: `probability_cache\control\volume_*.npz`.
- Recall-loss cache: `probability_cache\recall_loss\volume_*.npz`.
- Cache field name is `probability`, not `prob`.
- Control checkpoint: `mark 1\mark_4_outputs\mark_4_best.pth`.
- Recall-loss checkpoint: `mark 1\mark_4c_outputs\recall_loss_best.pth`.

Always recompute and record checkpoint SHA-256 in a freeze/confirmation phase. Do not copy a hash from prose without verifying the file currently on disk.

## Known code and metric corrections

- Mark 4C control patient CSV used `micro_dice`; visualization expected `dice`. The notebook was patched to rename the column.
- Mark 4C experimental-arm mean patient Dice initially included tumour-empty patients. Mark 4D corrected the population to nine tumour-positive patients.
- Mark 4D cache files use `probability`; reader code was patched from the incorrect `prob` key.
- The latest Mark 4D and Mark 4E executed notebooks contain no saved errors.

## Historical documentation caveat

`understanding the project/` is valuable for the earlier Practice timeline, parameters and fixed data issues, but its stated current plan predates Mark 1 through Mark 4E. Use it for history, not current status. The current decision is defined by this folder plus Mark 4D/4E machine-readable artifacts.

## Reproducibility checklist for every new phase

- Create `step_NN_short_descriptive_name\` under `mark 1 (part 2)\`.
- Keep its notebook, README, outputs, patches and error record inside that folder.
- Verify manifest hash.
- Record notebook path and execution date.
- Record all source artifact paths and hashes.
- Record software versions and CUDA/device information when relevant.
- Set and record seeds.
- Save complete parameters in JSON.
- Validate sample, volume and slice alignment.
- Save complete CSVs and bounded figures.
- Save a machine-readable gate.
- Assert test lock.
- Preserve existing outputs; create a new phase output directory rather than overwriting evidence.
- Save the phase parameter record, probability contract and result classification required by `07_SUCCESS_CRITERIA_PARAMETERS_AND_OUTPUTS.md`.

## New-work path convention

All future artifacts use this layout:

```text
mark 1 (part 2)/
  PROJECT_STATUS.md
  step_NN_short_descriptive_name/
    README.md
    step_NN_short_descriptive_name.ipynb
    outputs/
      configuration.json
      provenance.json
      expected_vs_actual.csv
      gate_result.json
      *.csv
      *.png
    error_fixes.md        # only when errors occur
    create_*.py           # optional notebook generator
    patch_*.py            # optional durable repair
```

No future phase may use an older Mark 1 output directory as its write destination.

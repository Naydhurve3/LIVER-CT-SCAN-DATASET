# Canonical Data Reference

`Practice/` is the authoritative evidence store for dataset observations and
experiment decisions. Source code and documentation must be reconciled to
these artifacts rather than silently replacing them.

## Source artifacts

- `Practice/lits_eda.ipynb`: executed EDA notebook and embedded outputs.
- `Practice/eda_data.txt`: full EDA log.
- `Practice/eda summary.txt`: concise EDA summary.
- `Practice/split_tumor_audit data.txt`: historical audit; corrected below.
- `Practice/sprint1_sweep data.txt`: tumor-weight sweep and rationale.
- `Practice/eda_plots/`: generated visual evidence.

## Verified storage geometry

Direct inspection on 2026-07-10 found:

- Images: 512 x 512, 8-bit PNG, stored as RGBA.
- Masks: 256 x 256, 8-bit PNG, stored as RGB.
- Training representation: one channel at 256 x 256, with nearest-neighbour
  mask resizing.
- Counts: 131 volumes and 58,638 matched image/mask slices.

The PNG images are already windowed. They do not retain calibrated HU values,
so a numeric HU window such as [-100, 400] is not physically valid for this
source. Framework preprocessing leaves HU windowing disabled for LiTS PNG.

## Label semantics

The upstream extracted files are named `segmentation-*_lesionmask_*.png`, and
the foreground occupies only about 0.12% of native mask pixels. The canonical
task is therefore **binary liver tumor/lesion segmentation**, not combined
liver-plus-tumor segmentation. Older documents using “liver+tumor foreground”
are historical and must not be used to define metrics or paper claims.

## Canonical findings

- Tumor-positive slices: 7,114 (about 12.1%).
- Zero-tumor volumes: 13/131.
- Native-mask mean tumor burden: 0.1211% per volume.
- Train/validation/test volumes: 104/13/14.
- Train/validation/test slices: 40,667/10,685/7,286.
- Tumor-positive slice rates: 12.0%/9.7%/16.4%.
- Native-mask mean burden: 0.1003%/0.0874%/0.3071%.

The sequential test split has materially greater tumor burden than train and
validation. Results must disclose this distribution shift.

## Split-audit correction

The historical audit divided pixels from 256 x 256 masks by a hardcoded
512 x 512 area. Its burden values are one quarter of the native-mask values.
At a native-mask low-burden threshold of 0.05%:

| Split | Zero | Low nonzero | Combined | Share |
|---|---:|---:|---:|---:|
| Train | 8 | 66 | 74 | 71.2% |
| Validation | 4 | 5 | 9 | 69.2% |
| Test | 1 | 5 | 6 | 42.9% |

Train and validation remain broadly similar. Test has fewer near-zero-burden
volumes and substantially higher burden. Keep the split for provenance, but
use volume-wise confidence intervals and state the limitation.

## Sprint 1 sweep decision

The one-epoch sweep supports carrying tumor sampler weight 3 into the pilot:
it had the best tumor Dice (0.0735) and near-best AUPRC (0.1646), while weight 5
continued the low-precision/high-recall over-prediction trajectory. These are
selection diagnostics, not final generalization results.

# Error fixes

## Repair 1 - incomplete liver-tumour label pattern

- Symptom: the first conversion found only 3 nonzero tumour cases although the official dataset description reports 15/20.
- Root cause: the regex required numbered folders such as `livertumor01`; many cases use `livertumor`, and case 19 uses `livertumors`.
- Repair: the frozen hepatic-tumour definition is now `^livertumors?\d*$`. This includes only the source liver-tumour family and deliberately excludes case 5 adrenal labels (`leftsurretumor`, `rightsurretumor`) and case 7 generic `tumor`.
- Preserved outputs: all 20 normalized case directories remain in `outputs/normalized/` and are deterministically overwritten and reverified on rerun.
- Resume: regenerate `step_14.ipynb`, then Run All from the first cell.

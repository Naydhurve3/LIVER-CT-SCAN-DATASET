# Error fixes

## 2026-08-05 — stale Jupyter launcher

- Error: `jupyter-nbconvert.exe` referenced a removed uv-managed Python executable.
- Root cause: the launcher metadata was stale; the notebook was not at fault.
- Repair: execute through the project `.venv` notebook client with the preserved `ds_gpu` site-packages on `PYTHONPATH`.
- Resume: restart and Run All, or use the direct notebook-client command recorded in the project handoff.

## 2026-08-05 — optional `tabulate` dependency absent

- Error: the final report cell failed at `DataFrame.to_markdown()` because `tabulate` is not installed.
- Root cause: a presentation-only optional dependency was used unnecessarily.
- Repair: patched the generator and notebook to embed the table as fenced CSV using built-in `DataFrame.to_csv()`; all earlier CSVs and figures were preserved.
- Resume: rerun the notebook from the start. This phase performs no inference and reuses sealed caches.

## 2026-08-05 — DICOM-to-frozen-grid in-plane axis mismatch

- Detection: the reconstructed accepted `livertumor` source mask for `ircadb_18` had zero overlap with the sealed prediction even though the sealed Dice is nonzero.
- Root cause: DICOM masks were stacked `(z, y, x)`, but Step 14 transposes them to NIfTI `(x, y, z)` and Step 16 resizes `(x, y)` slices. Step 18 initially resized raw `(y, x)` slices without the required transpose.
- Repair: patched `resize_zyx()` in the generator and notebook to transpose every source slice before nearest-neighbour resizing.
- Preservation: Step 13 source files, Step 14 normalized volumes, Step 16 caches/metrics, and all sealed signatures remain unchanged. Step 18 outputs are regenerated because the earlier concordance measurements were invalid.
- Resume: restart the Step 18 kernel and Run All from Cell 1.

## 2026-08-05 — broad container mask obscured the diagnostic figures

- Detection: after geometry correction, `skin` explained nearly all predictions and was selected for the panels, obscuring the tumour-semantic comparison.
- Root cause: the generic highest-overlap selection did not distinguish broad container labels from the specific annotations being adjudicated.
- Repair: excluded `skin` and `liver` from the highest-specific-label summary/dashboard and explicitly selected `tumor`, `metastasectomie`, and `livertumor` for cases 7, 14, and 18 panels.
- Resume: restart and Run All; no inference is performed.

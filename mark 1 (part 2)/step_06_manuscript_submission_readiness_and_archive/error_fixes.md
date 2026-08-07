# Error fixes

## 5 August 2026 — generator quoting repair

- Initial static parsing found nested double quotes in the scientific-content status print statement.
- Patched the actual generator and regenerated the notebook with unambiguous quoting.
- No analysis cell had executed, no output was lost, and the sealed test state was unaffected.

## 5 August 2026 — ledger filename repair

- The first artifact-only execution stopped before analysis because the notebook expected `one_time_test_run_ledger.json`; the actual sealed Step 04 artifact is `run_ledger.json`.
- Patched the generator and notebook references to the verified filename.
- Resume by restarting the notebook and using **Run All**. No test source was accessed and no inference ran.

## 5 August 2026 — archive hashing runtime repair

- The first full audit reached archive hashing but exceeded the safe runtime while re-hashing multi-gigabyte probability-cache/model binaries.
- Patched the generator to hash contracts, code, reports, tables, figures and gates, while explicitly excluding bulky `.npy`, `.npz`, `.pt`, `.pth` and probability-cache payloads.
- Those excluded binaries remain integrity-covered by the existing signed Step 03–05 checksum inventories; their count and total bytes are recorded in `archive_manifest_summary.json`.
- Resume with **Restart Kernel and Run All**. Previously completed small audit CSVs may be safely overwritten; sealed upstream outputs are read-only.

# Legacy Checkpoint Registry

Existing checkpoints are preserved as historical artifacts. They were trained
before the canonical PNG preprocessing correction and are not directly
comparable with research-validation runs.

| Directory | Profile | Status |
|---|---|---|
| `s9_pilot/` | `legacy_simulated_hu_v1` | Has history and limited test metrics |
| `s9_pilot_v2/` | `legacy_simulated_hu_v1` | Degenerate foreground-overprediction run |
| `s9_pilot_v3/` | `legacy_simulated_hu_v1` | Has test metrics; calibration is unreliable |
| `s9_finetune_v4/` | `legacy_simulated_hu_v1` | Checkpoints present; final metrics/history absent |

`legacy_simulated_hu_v1` means the already-windowed 8-bit PNG was passed
through the former simulated HU transform. Re-evaluation must explicitly use
that historical transform and identify results as legacy. Canonical runs use
`png_8bit_no_hu_no_clahe_v1`.

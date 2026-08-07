# Error fixes

## 2026-08-06 — superseded Step 16 preflight signature

- Error: terminal signature assertion found two mismatches in Step 16 `preflight_signature.json` for `authorization_state.json` and `gate_result.json`.
- Root cause: the preflight signature captured the authorization-required state before the explicitly authorized one-time run. The final authorized execution intentionally replaced those files and produced `step_16_signature.json`; the preflight snapshot is historical, not the current signature authority.
- Repair: Step 21 now records raw `hash_matches`, marks the Step 16 preflight mapping as `superseded_preflight`, and requires `effective_pass` for either a current hash match or an explicitly superseded preflight entry. All non-superseded mappings must still match exactly.
- Outputs preserved: the initial gate/artifact inventories and signature detail CSVs remain available but are regenerated on the corrected Run All.
- Resume: restart the Step 21 kernel and Run All from Cell 1. No inference or medical source data are accessed.

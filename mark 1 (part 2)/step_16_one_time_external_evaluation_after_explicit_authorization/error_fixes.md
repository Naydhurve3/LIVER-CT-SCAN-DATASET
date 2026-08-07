# Error fixes

No Step 16 runtime repairs recorded yet.

## Pre-run integrity repair 1 - final ledger/signature ordering

- Risk found during authorization review: the first draft signed `run_ledger.json` and then modified the ledger to `sealed_complete`, which would invalidate the signed ledger hash.
- Repair: the final ledger is now sealed before the Step 16 signature is computed and is not modified afterward.
- Also repaired `authorization_state.json` so an authorized completed run records model loading and inference as true instead of retaining preflight values.
- No inference had occurred and no probability caches existed when this repair was made.

## Authorization record

- Exact authorization accepted: `I authorize the one-time Step 16 3D-IRCADb-01 external evaluation under the frozen Step 15 contract.`
- Recorded UTC: `2026-08-05T14:49:18.3333060Z`
- One-time run UUID: `61d1f140-c8b4-436a-928a-1e4b6f7c0b56`

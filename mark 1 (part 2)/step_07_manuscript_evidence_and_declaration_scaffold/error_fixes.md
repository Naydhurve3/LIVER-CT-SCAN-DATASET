# Error fixes

## 5 August 2026 — JSON boolean normalization

- The first full execution completed the evidence tables, manuscript copy, templates and figure, then stopped while serializing the final gate because a pandas/NumPy boolean is not directly JSON serializable.
- Patched the generator to normalize every requirement result to a native Python `bool` before saving `expected_vs_actual.csv` and `gate_result.json`.
- Regenerated the actual notebook. Existing completed outputs are deterministic and may be safely overwritten.
- Resume with **Restart Kernel and Run All**. No model, loader, test source or inference was accessed.

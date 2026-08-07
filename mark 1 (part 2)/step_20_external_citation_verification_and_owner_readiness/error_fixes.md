# Error fixes

## 2026-08-06 — embedded BibTeX delimiter collision

- Error: generator `SyntaxError: invalid decimal literal` at the BibTeX title.
- Root cause: the inner raw triple-single-quoted BibTeX string prematurely closed the outer triple-single-quoted notebook cell source.
- Repair: changed the inner BibTeX block to a raw triple-double-quoted string in the generator.
- Outputs preserved: none existed; generation failed before notebook creation.
- Resume: rerun `create_step_20_notebook.py`, then validate and Run All from Cell 1.

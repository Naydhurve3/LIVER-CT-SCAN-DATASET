# Error fixes

## Repair 1 - generator quote delimiter

- Symptom: the Step 17 generator raised a Python syntax error before writing the notebook.
- Root cause: the generated Markdown data card and its enclosing code-cell string both used triple single quotes.
- Repair: changed only the inner data-card delimiter to triple double quotes.
- Step 16 evidence, probability caches, ledger and signature were untouched.
- Resume: rerun the Step 17 generator, then execute `step_17.ipynb` from the first cell.

## Repair 2 - visual interpretation boundary

- Visual QA showed that the largest negative-control predictions overlap visible low-attenuation structures, while the accepted hepatic-tumour truth is empty.
- The data card now explicitly prohibits interpreting these predictions as biological hallucinations without expert/source-annotation review.
- The reliability chart is now labeled as all-pixel and background-dominated so its ECE cannot be mistaken for clinical calibration.
- This is interpretation/presentation QA only; no Step 16 result, cache, threshold or metric was changed.

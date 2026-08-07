# Step 16 — One-time external evaluation after explicit authorization

Open the short notebook `step_16.ipynb` from this folder.

The default configuration performs a safe signed preflight only and must end at `EXTERNAL_EVALUATION_AUTHORIZATION_REQUIRED`. It verifies the Step 15 signature, three checkpoint hashes, complete 20-case cohort, all 60 normalized paths, fusion, threshold, post-processing and local-test boundary without loading a model or running inference.

Do not enable the expensive cells unless the user supplies the exact authorization sentence stored in the Step 15 gate. After authorization, the generator and notebook must record the sentence, timestamp and a new UUID. A complete result may not be tuned or rerun.

## Verified safe preflight — 5 August 2026

- Result level: `EXTERNAL_EVALUATION_AUTHORIZATION_REQUIRED`; this is the expected successful default state.
- Passed 12/12 prerequisite checks: signed Step 15 contract, current signature, three checkpoint hashes, 20 unique patients, 15 positive and five negative patients, all 60 normalized paths, manifest identity, threshold 0.70, maximum fusion, no post-processing and local-test exclusion.
- Created a signed preflight package under `outputs/` and an empty `outputs/probability_cache/` directory.
- Model loaded: false. Run ledger created: false. Probability files written: zero. Inference performed: false. Local LiTS test accessed: false.
- Validation: both notebook names pass `nbformat`; every code cell parses; the short notebook executed top-to-bottom with zero errors.
- Next action: provide the exact authorization sentence. Then patch both the generator and notebook with the authorization, UTC timestamp and a new UUID before the single expensive Run All.

## Authorized one-time completion — 5 August 2026

- Exact authorization was recorded at `2026-08-05T14:49:18.3333060Z`; run UUID `61d1f140-c8b4-436a-928a-1e4b6f7c0b56`.
- Result level: `EXTERNAL_EVALUATION_COMPLETE`; all 11/11 predeclared mandatory rows passed.
- Global Dice: 0.847977. Mean positive-patient Dice: 0.711196 (95% patient-bootstrap CI 0.567803–0.827823). Median: 0.835514; minimum: 0.012992.
- Q1 positive-slice detection: 50.00%; positive predicted-empty rate: 9.86%; empty-slice false-positive rate: 11.57%.
- Generated and sealed 20 patient probability caches covering 2,823 slices. Sample coverage, unique patient identity, finite probabilities and checkpoint integrity were all 100%.
- The run ledger is `sealed_complete`; `rerun_permitted` is false. Step 16 signature combined SHA-256: `4fb3d6d14010a982912c9ca8e85854e326960fa4a60543da47a378c6fd1b13aa`.
- Important caveats: worst positive case `ircadb_18` Dice = 0.012992; smallest-lesion detection = 63.64%; all five tumour-negative cases had at least one predicted pixel.
- Local LiTS test accessed: false. No threshold sweep, tuning, post-processing change or case exclusion occurred.
- Next action: use Step 17 for read-only evidence validation and caveat-preserving data-card generation. Never rerun Step 16.

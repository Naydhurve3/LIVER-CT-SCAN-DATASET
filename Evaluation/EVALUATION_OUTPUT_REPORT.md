# Evaluation Folder — Output Analysis Report

> **Prepared**: 11 August 2026
> **Scope**: `Evaluation/output/` full analysis + relationship of the Evaluation folder to `mark 1 (part 2)/`.
> **Data sources**: all `*_gate_result.json`, CSVs, figures, caches and the `artifact_index.json` under `Evaluation/output/`.

---

## 1. Does the Evaluation folder use `mark 1 (part 2)` data/code logic?

**Short answer: No — and it doesn't need to.** The data/code relationship is the *opposite* direction:

```
Dataset build (build_corrected_20260713_214847_v2)
   └─► Practice/  (source checkpoint: multitask_best.pth)
   └─► mark 1/    (frozen caches, ROI manifests, checkpoints, histories)
          └─► Evaluation/  (notebooks 00–09 reproduce Mark 1 → Mark 4E)
                  └─ outputs written to Evaluation/output/
                  └─ artifacts MIRRORED into mark 1/mark_*_outputs/
                         └─► mark 1 (part 2)/  ← DOWNSTREAM CONSUMER
```

### 1.1 Evidence from the code

1. **Every Evaluation notebook's shared setup cell** (verified in `00_pipeline_overview_and_setup.ipynb`) reads only from:
   - `DATASET_ROOT = ...\Dataset\Liver\02_staging\build_corrected_20260713_214847_v2` (the canonical build)
   - `MARK1_DIR = PROJECT_ROOT / "mark 1"` (frozen artifacts)
   - `SOURCE_CHECKPOINT = Practice/multitask_liver_tumor_outputs/multitask_best.pth`
   - framework modules `src/framework/data/manifest_dataset.py`, `MobileNetV2UNet`, `FocalDiceLoss`
   - **There is no path pointing into `mark 1 (part 2)/` anywhere in the notebook code.**

2. **The only mentions of `mark 1 (part 2)` in the Evaluation folder are:**
   - `PROJECT_EVALUATION_DATA.md` — the *master data card (documentation)*, which cites part 2's data cards/reports as *sources of facts* (e.g., Step 04 test result Dice 0.767696, Step 16 external Dice 0.847977).
   - Comments in notebooks 03/07/08 identifying part 2 steps as **external consumers** of the mirrored artifacts (`training_roi_manifest.csv`, `mark_4_best.pth`, `recall_loss_best.pth`, `mark_4d_gate_result.json`, `mark_4e_gate_result.json`).
   - `README.md` mirror table noting the same consumers.

3. **The actual dependency is: Evaluation (and `mark 1/`) → `mark 1 (part 2)`.** Part 2's steps 01/02/03/15 read the *mirrored* outputs that Evaluation writes into `mark 1/mark_*_outputs/`. This is confirmed in the code: notebooks 02–08 end with a "PUBLISHED n artifacts to `mark 1/mark_*_outputs/`" step precisely so that part 2's hardcoded input paths see fresh artifacts.

### 1.2 Do we need to integrate part 2 now?

**No — nothing is missing.** Part 2's own results (test evaluation, external evaluation, fusion freeze, manuscript steps) are *separate phases with their own `outputs/` folders* inside each `step_XX/`, and their headline values are already consolidated into `PROJECT_EVALUATION_DATA.md` at the documentation level. If machine-readable part 2 gate JSONs should also be mirrored into `Evaluation/output/`, that would be an optional new consolidation step — not a repair of a broken dependency.

---

## 2. Proper Report of `Evaluation/output/`

### 2.1 Overall inventory

| Area | Count | Notes |
|---|---|---|
| Phase folders | 11 (`00`–`10` + registry) | one per notebook plus the registry consolidator |
| Data files (CSV/JSON) | **65** | 8+7+10+8+7+7+6+6+3 + hub 6 + registry 2 |
| Figure files (PNG) | **89** | incl. a `*_summary_dashboard.png` per phase + hub 49 |
| Probability caches (.npz) | 52 | Mark 1: 13, Mark 4B: 13, Mark 4D: 26 |
| Checkpoints (.pth) | 5 | 3 overfit arms (Mark 3) + 2 ablation arms (Mark 4C) |
| `artifact_index.json` | **211 entries** | complete manifest (version 2) after notebook 11 consolidation — see caveat §2.3 |

**Run provenance:** artifact timestamps run `2026-08-10T19:28` → `2026-08-11T06:42` UTC — a single full sequential pass completed this morning (`reproduction_verification.json` generated `2026-08-11T06:42:22`).

### 2.2 Phase-by-phase gate results (the core story)

| Phase | Status | Headline result | Gate |
|---|---|---|---|
| **Mark 1** | `diagnostic_complete` | Signal **mislocalized/absent** | ❌ calibration failed |
| **Mark 2** | `feasibility_complete` | 100% tumor containment, median crop 42.7% | ✅ |
| **Mark 3** | `overfit_pass` | Hard micro-Dice **0.9006** (17 ep, 1ch) | ✅ |
| **Mark 4** | `smoke_fail` | 5/6 targets — pos-empty 36.85% (>35%) | ❌ |
| **Mark 4B** | `diagnostic_complete` | Threshold tuning can't fix recall (5/6) | ❌ |
| **Mark 4C** | `ablation_fail` | Two-channel collapsed; recall-loss improved recall but killed V116 | ❌ |
| **Mark 4D** | `diagnostic_complete_no_full_pass` | V116 is a **localization failure**, not ROI clipping (5/6) | ❌ |
| **Mark 4E** | `fusion_pass` | **max-fusion @ 0.70 passes all 6 targets** | ✅ |
| **09 Consolidated** | — | 8/8 gates reproduced, **56/56 comparisons, worst diff = 0.0** | ✅ |

#### Mark 1 (`01_mark_1/`)
Best observed config: threshold 0.70, raw mode → global Dice 0.4537, precision 0.727, recall 0.330, mean patient Dice 0.333, **V104 ≈ 0 and V116 ≈ 0** (≈1e-11), Q1 detection 27.4%, 46.6% positive predicted-empty, 3.2% empty-FP. Bootstrap 95% CI for mean patient Dice: 0.134–0.512. HU-contrast table shows V104 is an extreme outlier (median contrast −89.5 HU, effect size −4.12). 5 figures incl. calibration-frontier dashboard and V104/V116 localization panels; 13 validation probability caches.

#### Mark 2 (`02_mark_2/`)
Selected ROI: liver threshold 0.50, padding 16, `largest_3d`; V104 & V116 containment **1.0**, min/mean positive-patient containment 1.0, median crop-area ratio 0.427 (max 0.533), 0 empty ROIs. Multi-window analysis: broad window has clean tumor/liver separation; the narrow window saturates >70% of V104 tumor pixels → explains the hypodense-suppression diagnosis.

#### Mark 3 (`03_mark_3/`)
Selected `broad_1ch`, 17 epochs, hard micro-Dice **0.900557**, 0% predicted-empty. (2ch: 0.9039 @ 20 ep; 3ch: 0.9005 @ 42 ep — no benefit worth the cost.) Training-ROI gate passed (containment 1.0, median crop 0.4204, roundtrip Dice 1.0). Outputs include 3 `.pth` checkpoints, overfit history, training ROI manifest.

#### Mark 4 (`04_mark_4/`)
5-epoch smoke: mean patient Dice 0.3639, V104 0.0672, V116 0.0108, Q1 42.6%, **pos-empty 36.85% FAIL**, empty-FP 3.42%. 5/6 continuation targets; **0/6 final (aspirational) targets**. Per-patient table shows the failure pattern: V104 65.6% and V116 94.0% predicted-empty.

#### Mark 4B (`05_mark_4b/`)
Full 0.05–0.70 threshold sweep: pos-empty stayed **36.3–37.0% at every threshold** → threshold-only calibration is insufficient. Selected 0.60: mean 0.3646, V104 0.0646, V116 0.0104, Q1 42.6%, pos-empty 37.0%, empty-FP 3.40%; bootstrap p2.5 0.189 / p50 0.369 / p97.5 0.531 (lower bound below target → report-only).

#### Mark 4C (`06_mark_4c/`)
3-arm ablation: **control** 0.3639 mean (5/6), **two-channel collapsed** (patient Dice ≈ 0, pos-empty 100%, 1/6 — rejected), **recall-loss** 0.2596 mean but V104 0.1038, Q1 47.9%, pos-empty 30.1% and **V116 0.00085 fail** (4/6). Training histories + 2 `.pth` checkpoints saved.

#### Mark 4D (`07_mark_4d/`)
Metric reconciliation: recall-loss @ 0.60 → mean 0.3766, V104 0.1002, Q1 47.9%, pos-empty 30.5%, empty-FP 4.79%, **V116 0.00071 FAIL**. V116 size-stratum table: detection ≈ 0% in Q1/Q2, ≤9% in Q3/Q4 for *both* models, median truth-region probability = 0 → **recognition failure, not ROI clipping** (100% of 152,763 pixels inside ROI).

#### Mark 4E (`08_mark_4e/`)
7 fusion policies × 14 thresholds. Selected **pixelwise maximum @ 0.70**: mean 0.3771, V104 0.1166, V116 0.0105, Q1 50.6%, pos-empty 27.4%, empty-FP 5.55% → **6/6 PASS**. (mean & weighted fusions also passed; `maximum` chosen by predeclared highest-mean-Dice rule). Decision: `FREEZE_FUSION_POLICY_AND_THRESHOLD`.

#### 09 Consolidated (`09_consolidated/`)
`unified_gate_summary.csv` joins all gates; `reproduction_verification.json`: **56/56 comparisons passed, worst abs diff 0.0, zero failures** → the Evaluation suite exactly reproduces the original `mark 1/` gates. Figures: cross-phase progress + pipeline status strip.

### 2.3 Caveats & observations

1. **`artifact_index.json` is now a complete SHA-256 manifest of every artifact under `output/`.** Notebook `11_artifact_registry_consolidation.ipynb` scans the output tree and registers all 211 files — 65 data CSVs/JSONs, 89 figures, 52 `.npz` caches and 5 `.pth` checkpoints (registry `version: 2`, `registry_verification.json` confirms 211/211 on-disk files, zero missing/stale). Earlier runs of notebooks 01–09 registered only figures because most data tables were written with raw `pandas.to_csv`/`json.dump` instead of the `save_table`/`save_json` helpers.
2. **V116 remains a knife-edge pass**: margin above the 0.01 floor is only ~0.00047 at Mark 4E (and 0.0105 here). Not robust-generalization evidence.
3. **Mark 2 & Mark 3 rows in `unified_gate_summary.csv` have empty metric columns** — expected (their gates don't use the 6-metric suite), just don't misread as missing data.
4. **`00_pipeline_overview/` is a skeleton** (0 data/figures) — by design.
5. **Overall story is consistent**: recall cannot be bought with thresholds or a second input channel; the only mechanism that cleared all 6 validation targets was checkpoint fusion — and even then V121 later failed the formal test acceptance (Dice 0.767696, documented in `PROJECT_EVALUATION_DATA.md` §13).

---

*End of report.*

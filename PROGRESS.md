# Project Progress Record — Liver CT Tumor Localization & Segmentation

> **Purpose**: Single source of truth for *what has been achieved, where the evidence lives, and what is frozen/finalized*. Read this before starting any new work so you inherit the state instead of re-deriving it.
> **Last updated**: 13 August 2026 (backup-zip metadata extraction + heavy-zip removal)

---

## 1. Project identity

- **Task**: Liver + liver-tumor segmentation and *localization* from CT volumes, with a strong emphasis on reproducible, gate-driven evaluation.
- **Data**: LiTS-17 (131 volumes), canonical build `build_corrected_20260713_214847_v2`, manifest SHA-256 `575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889`.
  - Patient-disjoint splits: **Train 104 / Val 13 / Test 14** volumes; 58,638 slices; 7,169 tumor-positive slices.
  - External evaluation set: **3D-IRCADb-01** (20 anonymized patients) — Steps 12–21 of part 2.
- **Environment snapshot**: Python 3.11.15 (`.venv`), torch 2.5.1+cu124, RTX 3050 Ti.
- **Framework**: `src/framework/` (manifest dataset loader, MobileNetV2UNet ~6.8M params, FocalDiceLoss, trainer, experiment harness).

---

## 2. Data pipeline (frozen facts)

- **Provenance chain**: `slice_manifest.csv` → corrected build → stratified splits → manifest dataset.
- **ROI crop protocol** (frozen, Mark 2): liver threshold 0.50, `largest_3d`, padding 16, output 256×256.
- **HU window**: `[-160, +240]`.
- **Forensics fixed during build**: 47 orientation flips corrected; 4× burden-denominator fix; mask semantic lock (liver vs tumor semantics enforced).
- **3D morphology**: 845 tumor components; Q1–Q4 size-stratification bins defined from train data.

---

## 3. Research milestones Mark 1 → Mark 4E (validation loop, ALL reproduced)

The `Evaluation/` notebook suite (00–12) is the **canonical runnable reproduction**. It re-derives every original `mark 1/` gate with drift < 1e-4 and mirrors artifacts back to `mark 1/mark_*_outputs/` for part-2 consumers.

| Phase | Notebook | Status | Headline result |
|---|---|---|---|
| Mark 1 | `01_mark_1` | `diagnostic_complete` | Frozen model mislocalizes/absents tumor signal; calibration cannot recover it (V104 ≈ V116 ≈ 0, ≈1e-11) |
| Mark 2 | `02_mark_2` | `feasibility_complete` | ROI (liver @0.50) contains 100% of tumor; median crop-area ratio 0.427 |
| Mark 3 | `03_mark_3` | `overfit_pass` | Capacity gate: hard micro-Dice **0.9006** (broad_1ch, 17 ep) |
| Mark 4 | `04_mark_4` | `smoke_fail` | 5-epoch smoke: 5/6 targets; pos-empty 36.85% (>35%) |
| Mark 4B | `05_mark_4b` | `diagnostic_complete` | Threshold sweep cannot fix recall (pos-empty 36.3–37.0% at all thresholds) |
| Mark 4C | `06_mark_4c` | `ablation_fail` | Two-channel collapses; recall-loss improves recall but kills V116 (0.00085) |
| Mark 4D | `07_mark_4d` | `diagnostic_complete_no_full_pass` | V116 is a **localization/recognition failure**, not ROI clipping (100% of pixels inside ROI) |
| Mark 4E | `08_mark_4e` | **`fusion_pass`** | **Pixelwise max-fusion @ 0.70 passes all 6 validation targets** — 6/6 |
| Consolidated | `09_consolidated` | ✅ | 8/8 gates reproduced; 56/56 comparisons, worst diff 0.0 |
| Registry | `11_artifact_registry` | ✅ | Complete SHA-256 manifest of every `output/` artifact (211 entries) |
| Visualization | `10_visualization_hub` | ✅ | Unified dashboard hub over all phases |

**The core scientific finding**: recall cannot be bought with thresholds or a second input channel. The only mechanism that cleared all 6 validation targets was **checkpoint fusion** `max(P_control, P_recall) @ 0.70`.

**V116 caveat**: the fusion pass is knife-edge (margin over the 0.01 floor ~0.00047 at 4E). Not robust-generalization evidence.

---

## 4. Formal test evaluation (one-time locked, Step 04)

- **Result**: global Dice **0.767696** on the locked test split.
- **Status**: formal acceptance **FAILED on V121** (pos-empty threshold) — documented with full failure analysis (V120, V127, lesion-size limitation).
- **Artifacts**: `mark 1 (part 2)/step_04_*/outputs/` (authorization record, bootstrap uncertainty, failure cases, test signature).

---

## 5. External evaluation — 3D-IRCADb-01 (Steps 12–21)

- **Step 16 one-time external eval**: global Dice **0.847977**.
- Caveat: source-label concordance variance (Step 18 adjudication) — treat with care.
- Full chain: ingestion QC (13) → normalized conversion + parity QC (14) → frozen contract (15) → eval (16) → evidence validation + data card (17) → label concordance (18) → manuscript integration (19) → citation verification (20) → terminal evidence chain + handoff (21).

---

## 6. Part 2 pipeline status (Steps 00–21)

- 21-step gated pipeline in `mark 1 (part 2)/`, each step has `outputs/` with `provenance.json`, `configuration.json`, `expected_vs_actual.csv`, `gate_result.json`, step signature.
- **Step 00 Model Improvement Program**: sealed 8-volume holdout; Phase 1 smoke gate — all arms failed, program **halted**.
- **Step 03 Final inference policy**: frozen (fusion policy). Step 21 terminal evidence chain + handoff complete.

---

## 7. Practice ablations (historical, superseded by Evaluation suite)

Frozen reference ablations in `Practice/` — each with `*_outputs/` folder:
baseline, patient-aware, patient/lesion-balanced sampler, recall-aware Focal-Tversky, stabilized composite loss, intensity robustness (domain forensics), 2.5D context ablation, 3D post-processing, multi-task liver+tumor, dataset-pairing forensics, unified dataset prep/EDA.
> These are inputs/reference only; do not extend them — the canonical suite is `Evaluation/`.

---

## 8. Research / manuscripts

- **UP³RE-Net / MedSegX track**: novelty claims + 6 patent claims; UWACL loss; FAUP-Net + UWACL-v2 designs (in-progress, `research/`, `papers/`).
- Manuscript pipeline (Step 06–11): submission-readiness gates, internal peer review, venue selection — status records in part-2 step outputs.

---

## 9. Structure & organization (this pass)

Backup snapshot of all generated outputs/code/notebooks/docs is stored under `backup/` (git-ignored), with a `MANIFEST.json` mapping every file to its zip and SHA-256. Re-run anytime with `python tools/backup_project.py`.

### Backup inventory (created 12 Aug 2026 — 1,995 files / 9.6 GB in 9 zips, all integrity-verified)

| Zip | Contents | Files | Size |
|---|---|---|---|
| `01_notebooks_all.zip` | every `.ipynb` (part2 41, Practice 24, Evaluation 14, notebooks 14, studies 10, mark1 9) | 112 | 51 MB |
| `02_code_all.zip` | every `.py` incl. 49 generators + tools/tests | 212 | 0.6 MB |
| `03_docs_md_yaml.zip` | all `.md`/`.yaml`/`.txt`/`.bib` docs | 181 | 7.9 MB |
| `04_output_evaluation.zip` | `Evaluation/output/` data+figures (caches/checkpoints live in 06/07) | 194 | 20.7 MB |
| `05_outputs_legacy_mirror.zip` | `mark_1_to_4e_outputs/`, `mark 1/mark_*_outputs/`, `Practice/*_outputs/`, root `outputs/results/figures` | 527 | 121 MB |
| `06_models_pth_all.zip` | all 80 `.pth` checkpoints | 80 | 5,767 MB |
| `07_caches_npz_all.zip` | all 216 `.npz` probability caches | 216 | 3,665 MB |
| `08_outputs_part2.zip` | part-2 step outputs (csv/json/png/signatures) | 454 | 11.4 MB |
| `09_other.zip` | misc small files (`.gitkeep`, env example, scripts) | 19 | ~0 MB |

- IRCADb external dataset (DICOM, `.zip`, `.vtk`, `.gz`) intentionally **not archived** (re-downloadable): 34,404 paths recorded in `backup/08_ircadb_REFERENCE.md`.
- Restore = unzip each zip at repo root; relative paths preserved.
- Before/after checks: every zip passed `testzip()` integrity; every file has a SHA-256 in `MANIFEST.json`.

**Canonical homes going forward:**
| Concern | Location |
|---|---|
| Code | `src/framework/`, `tools/`, `scripts/`, `tasks/`, `app/`, `deployment/` |
| Research hub | `Evaluation/` (notebooks 00–12 + `output/`) |
| Frozen part-2 | `mark 1 (part 2)/` (steps 00–21) |
| Frozen part-1 | `mark 1/`, `Practice/` (read-only reference) |
| External data | `data/external/` |
| Docs | `docs/` (incl. merged `understanding the project/` knowledge base, moved 14 Aug 2026) |

---

## 10. Run & verify commands

- Notebooks: `Evaluation/00_pipeline_overview_and_setup.ipynb` → run 01–12 in order (`.venv` kernel; REUSE mode default for fast deterministic re-runs).
- Tests: `pytest` from repo root.
- Framework CLI: `python run.py` / `tools/train.py`.
- Regeneration of deleted heavy outputs: `docs/REPRODUCTION_AND_REGENERATION.md`.

---

## 11. Next actions (open items)

1. **DONE (this pass)**: full compressed backup created (`backup/`, 9.6 GB / 1,995 files, SHA-256 manifest, integrity-verified); regenerable caches (`__pycache__`, `.pytest_cache`, `.pytest-tmp`) removed; empty root `output/` removed; dead runtime files moved to `archive/`; `data/external/` documented; `PROGRESS.md` + `data/README.md` + `archive/README.md` added; test suite 243/243 green.
2. **DONE (this pass) — duplicate pruning** (all items verified present in `backup/MANIFEST.json` before deletion, SHA-256 tracked):
   - Removed `Evaluation/mark_1_to_4e_outputs/` legacy mirror (**1,299 MB / 145 files**) — `output/` is the single canonical source (251 files verified intact, incl. `artifact_index.json`).
   - Removed 8 redundant `*_last.pth` in `Practice/` (**629 MB**) — each has its `*_best.pth` counterpart kept.
   - Removed 2 zero-reference dataset-prep build snapshots `Practice/unified_dataset_preparation_outputs/20260713_213247` + `20260713_214500` (**42 MB / 33 files**) — canonical `20260713_214847` (102 refs) + `20260724_211625` + `20260713_214524` kept.
   - **Total freed ~1.9 GB.** Root `outputs/`, `results/`, `figures/` kept (small, ~22 MB, active write targets).
3. **Kept by design (not duplicates):** the `Evaluation/output` ↔ `mark 1/mark_*_outputs/` mirror (~1.36 GB, 138 identical groups) — `mark 1/mark_*_outputs/` is the hardcoded input for part-2 steps, so both copies are required. `Practice/eda_plots` ↔ `figures/` PNGs and `continuation_process_map.png` are referenced by notebooks/docs.
4. **DONE (this pass) — regenerable-output removal (all 222 files verified in `backup/MANIFEST.json` before deletion, SHA-256 tracked; `csv/json/png` results untouched)**:
   - Removed **164 `.npz` probability caches (~2,461 MB)** — `Evaluation/output`, `mark 1`, `mark 1 (part 2)` steps 02/04/16, `Practice` — all recomputed automatically by their notebooks when missing.
   - Removed **58 superseded `.pth` checkpoints (~4,613 MB)** — `models/s9_pilot*` + `s9_finetune_v4` (3,293 MB), `models/research_validation` (329 MB), `experiments/sprint1` sweeps (741 MB), `Practice` non-multitask `*_best.pth` (659 MB), `mark 1/mark_3_outputs` overfits + `mark_4_last.pth` (111 MB).
   - **Kept the 4 required input checkpoints** (`multitask_best.pth`, `mark_4_best.pth`, `recall_loss_best.pth`, `two_channel_best.pth` + mirrors) — deleting them would break all Evaluation + part-2 notebooks.
   - **Total freed in this pass ~7.07 GB** (cumulative across all passes ~9.0 GB).
   - **Regeneration recipes** (which code recreates each deleted artifact, in what order) recorded in `docs/REPRODUCTION_AND_REGENERATION.md`. Instant recovery = unzip from `backup/` via `MANIFEST.json`.
5. **DONE (this pass) — step_13 raw-DICOM removal (~11.3 GB freed)**:
   - Removed `mark 1 (part 2)/step_13_3d_ircadb_ingestion_and_qc/outputs/extracted/` (34,322 files / **~10.8 GB**) — raw 3D-IRCADb-01 DICOM, fully re-extractable offline from the 20 verified patient ZIPs kept in `outputs/raw_downloads/` (782 MB, sha-256 in `download_manifest.csv`, all `testzip()`-clean).
   - Removed stale incomplete download `3Dircadb1.zip.part` (**820 MB**, absent from the manifest).
   - Downstream steps 15–21 read only step_14's **normalized `.nii.gz`** (frozen), not the raw DICOM — no downstream dependency.
   - All source evidence (`extracted_file_inventory.csv`, QC/gate/signature csv/json/png, ~2 MB) kept; re-extraction recipe in `step_13_3d_ircadb_ingestion_and_qc/README.md` and `docs/REPRODUCTION_AND_REGENERATION.md` §3.7.
   - **Cumulative freed across all passes ~20.3 GB.**
6. **DONE (this pass) — heavy backup-zip removal (~8.8 GB freed)**:
   - Removed `backup/06_models_pth_all.zip` (**5,500 MB**, 80 `.pth`) and `backup/07_caches_npz_all.zip` (**3,495 MB**, 216 `.npz`) after extracting full metadata to `docs/BACKUP_CHECKPOINT_CACHE_INVENTORY.md` (every file: path, size, SHA-256).
   - All 9 **required checkpoints remain in the working tree** (verified) — the zips held only redundant copies of those plus already-deleted regenerable training/cache artifacts.
   - `backup/` now **0.2 GB**: notebooks, code, docs, evaluation outputs, legacy-mirror outputs, part-2 outputs, IRCADb reference + updated `MANIFEST.json` (1,699 files).
   - **Cumulative freed across all passes ~29.1 GB.**
7. **DONE (this pass) — environment/cache removal + git gc + light reorg (~12.2 GB freed)**:
   - Removed `.uv-cache/` (**5,416 MB**), `.uv-python/` (**73 MB**), and `.venv/` (**5,411 MB**, Python 3.11.15). Environment is fully recreatable: `uv venv` + `uv pip install -r requirements.txt`, or `conda env create -f environment.yaml`. `requirements.txt`/`environment.yaml` kept at root.
   - Removed 646 `__pycache__/` dirs (**113 MB**) + `.pytest_cache` + `.pytest-tmp` + empty `.agents/`, `.codex/`, root `output/`.
   - Removed `step_14_3d_ircadb_normalized_conversion_and_parity_qc/outputs/normalized/` (**60 NIfTI / 923 MB**) — external-eval evidence (Steps 17–21) reads only the remaining csv/json/png/signatures; per-file SHA-256 preserved in `outputs/conversion_manifest.csv`. Regenerate by re-running `step_14.ipynb` against the kept step_13 zips (`outputs/raw_downloads/`), verified byte-for-byte against the manifest hashes. Recipe also in `step_14.../README.md` and `docs/REPRODUCTION_AND_REGENERATION.md`.
   - `git gc --prune=now --aggressive`: `.git` **682 MB → 16 MB** (removed 292 MB+ of unreachable blobs + 8,643 dangling objects; `git fsck --full` clean, history intact).
   - Light reorg: `dataset.md` + `LITS_DATASET_EDA_GITHUB_CARD.md` moved to `docs/`; `understanding the project/` (11 files) merged into `docs/understanding the project/`; link in `notebooks/README.md` and structure tree in `README.md` updated; `docs/README.md` index added; `outputs/`/`results/`/`figures/` each got a one-line README.
   - **Cumulative freed across all passes ~41 GB.** Working tree now ~1.1 GB of project data (incl. ~0.6 GB part-2 evidence + required checkpoints).

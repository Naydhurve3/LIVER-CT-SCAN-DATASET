# Version History — Liver Tumor Localization & Segmentation (LiTS-17)

This document tells the **progress story** of the project across versions. Each version has its own hub with rich detail, links to the real notebooks / figures / outputs, and a goal → did → got structure.

| Version | Focus | Status | Hub |
|---|---|---|---|
| **[v1 — Phase 1 Research](versions/v1/VERSION.md)** | EDA, spatial forensics, patient splits, Mark 1 → 4E checkpoint fusion | ✅ Complete (original repo snapshot) | [`versions/v1/`](versions/v1/VERSION.md) |
| **[v2 — Phase 2](versions/v2/VERSION.md)** | Reproducible Evaluation suite, part-2 steps 00–21, 3D-IRCADb external validation, storage recovery | 🟢 Active (current) | [`versions/v2/`](versions/v2/VERSION.md) |

---

## The journey at a glance

```
LiTS-17 raw (131 CT volumes)
   │  ├─ v1: acquire → hash → canonical build → 47 flip repairs → splits → Mark 1→4E fusion
   │  └─ v2: Evaluation/00–12 reproduces all of v1 → part-2 step_00–21
   │        └─ 3D-IRCADb-01 external validation → evidence chain → lean regenerable repo
```

### v1 → what changed
- **Before v1**: no verified build, hidden orientation/denominator/label artifacts.
- **After v1**: corrected canonical build, patient-disjoint splits, 2-stage ROI pipeline, Mark 4E fusion passing all 6 gates (Dice 0.3771, Q1 50.57%, ES-FP 5.55%).

### v2 → what changed
- **Reproduction**: every v1 result formally re-run and verified via `Evaluation/00–12`.
- **External validation**: generalization measured on 3D-IRCADb-01 (20 patients) through a frozen, one-time, evidence-backed contract (steps 12–21).
- **Governance**: `PROGRESS.md` as the single source of truth; test suite 243/243.
- **Lean repo**: ~41 GB of caches / dead weights removed, every artifact regenerable via `docs/REPRODUCTION_AND_REGENERATION.md`.

---

## Version hubs (rich detail, direct links)

- [**Version 1 hub**](versions/v1/VERSION.md) — goal, what we did, what we got, notebook/doc/figure links, key images
- [**Version 2 hub**](versions/v2/VERSION.md) — goal, what we did, what we got, Evaluation/step links, key images

---

## How versions are structured

- Each version folder contains `VERSION.md` (the story) + `assets/` (small copies of headline images).
- All heavy content (notebooks, outputs, evidence) lives in its **real** location; version hubs **link** to it — nothing is duplicated.
- The current working state is always **v2**. Any future work should continue from v2.

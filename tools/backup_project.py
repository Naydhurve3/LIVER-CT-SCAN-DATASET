from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "backup"
EXCLUDE_DIRS = {
    ".venv", ".git", ".uv-cache", ".uv-python", "__pycache__",
    ".pytest_cache", ".pytest-tmp", ".ipynb_checkpoints", "backup",
    ".eggs", ".egg-info",
}
EXCLUDE_EXT_ARCHIVE = {".zip", ".gz", ".vtk"}
TEXT_EXT = {".md", ".yaml", ".yml", ".txt", ".tex", ".bib", ".html", ".ipynb", ".py", ".json", ".csv", ".png", ".jpg", ".jpeg", ".log"}


def classify(rel: str, ext: str) -> str:
    parts = rel.split("/")
    if ext == ".pth":
        return "pth"
    if ext == ".npz":
        return "npz"
    if ext == ".ipynb":
        return "notebook"
    if ext == ".py":
        return "code"
    if ext in EXCLUDE_EXT_ARCHIVE or rel.endswith(".nii"):
        return "external"
    if ext == "" and ("3dircadb" in rel.lower() or "extracted" in rel.lower()):
        return "external"
    if rel.startswith("Evaluation/output") and parts[1] == "output":
        return "output_eval"
    if rel.startswith("Evaluation/mark_1_to_4e_outputs/"):
        return "outputs_legacy"
    if rel.startswith("mark 1/mark_"):
        return "outputs_legacy"
    if rel.startswith("Practice/") and len(parts) > 1 and "_outputs" in parts[1]:
        return "outputs_legacy"
    if rel.startswith("outputs/") or rel.startswith("results/") or rel.startswith("figures/"):
        return "outputs_legacy"
    if parts[0] == "mark 1 (part 2)":
        if "outputs/extracted" in rel or "outputs/raw_downloads" in rel:
            return "external"
        return "part2"
    if ext in TEXT_EXT:
        return "docs"
    return "other"


ZIP_NAMES = {
    "notebook": "01_notebooks_all.zip",
    "code": "02_code_all.zip",
    "docs": "03_docs_md_yaml.zip",
    "output_eval": "04_output_evaluation.zip",
    "outputs_legacy": "05_outputs_legacy_mirror.zip",
    "pth": "06_models_pth_all.zip",
    "npz": "07_caches_npz_all.zip",
    "part2": "08_outputs_part2.zip",
    "other": "09_other.zip",
}
STORED_CATS = {"pth", "npz", "other"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    BACKUP.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(list)
    external_note = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(ROOT).parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        ext = path.suffix.lower()
        cat = classify(rel, ext)
        if cat == "external":
            external_note.append(rel)
            continue
        buckets[cat].append(path)

    manifest = {"version": 3, "root": str(ROOT), "created": None, "files": {}}
    summary = []
    for cat, zip_name in ZIP_NAMES.items():
        files = buckets.get(cat, [])
        if not files:
            continue
        zip_path = BACKUP / zip_name
        mode = zipfile.ZIP_STORED if cat in STORED_CATS else zipfile.ZIP_DEFLATED
        total = 0
        with zipfile.ZipFile(zip_path, "w", compression=mode, allowZip64=True) as zf:
            for path in files:
                rel = path.relative_to(ROOT).as_posix()
                zf.write(path, arcname=rel)
                total += path.stat().st_size
        size_mb = round(zip_path.stat().st_size / 1e6, 1)
        summary.append((zip_name, len(files), size_mb))
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            manifest["files"][rel] = {
                "zip": zip_name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        print(f"{zip_name}: {len(files)} files -> {size_mb} MB")

    manifest_path = BACKUP / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ref = BACKUP / "08_ircadb_REFERENCE.md"
    ref.write_text(
        "# IRCADb External Dataset — Reference (not backed up)\n\n"
        "Re-downloadable external data (3D-IRCADb-01 + LiTS17 raw archives). Skipped from backup by design.\n\n"
        f"Recorded {len(external_note)} file paths (hashes omitted for 12 GB raw data).\n"
        "Sources: IRCADb-01 (https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) "
        "and the download zips listed under step_13 outputs/raw_downloads.\n",
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    total_files = sum(n for _, n, _ in summary)
    print(f"{total_files} files across {len(summary)} zips -> {sum(s for _, _, s in summary)} MB")
    print("IRCADb external files recorded (not archived):", len(external_note))


if __name__ == "__main__":
    main()

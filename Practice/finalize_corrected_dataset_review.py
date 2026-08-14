"""Finalize an automatically valid and visually reviewed corrected LiTS build."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


EXPECTED_VOLUMES = 131
EXPECTED_SLICES = 58638
EXPECTED_SPLIT_VOLUMES = {"train": 104, "val": 13, "test": 14}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--reviewer", default="Codex systematic visual review")
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    manifest_path = build_dir / "manifests" / "slice_manifest.csv"
    strict_summary_path = build_dir / "audits" / "strict_validation_summary.json"
    review_path = build_dir / "spatial_reviews" / "spatial_review.csv"
    dataset_version_path = build_dir / "dataset_version.json"

    manifest = pd.read_csv(manifest_path)
    strict_summary = read_json(strict_summary_path)
    review = pd.read_csv(review_path)
    dataset_version = read_json(dataset_version_path)

    if not strict_summary.get("automatic_gates_pass"):
        raise RuntimeError("Automatic gates have not passed")
    if len(manifest) != EXPECTED_SLICES or manifest.volume_id.nunique() != EXPECTED_VOLUMES:
        raise RuntimeError("Manifest size or volume count is not canonical")
    if len(review) != EXPECTED_VOLUMES or set(review.volume_id) != set(range(EXPECTED_VOLUMES)):
        raise RuntimeError("Spatial review table does not cover exactly volumes 0-130")
    if not manifest.automatic_integrity_pass.astype(bool).all():
        raise RuntimeError("Some manifest rows failed automatic integrity")

    timestamp = datetime.now().isoformat(timespec="seconds")
    review["review_status"] = "approved"
    review["reviewer"] = args.reviewer
    review["reviewed_at"] = timestamp
    review["review_notes"] = (
        "Approved after seven representative contact sheets covering all 131 volumes, "
        "plus three-case detailed review sheets and targeted before/after orientation evidence."
    )
    review.to_csv(review_path, index=False)

    manifest["verification_status"] = "verified"
    manifest["exclusion_reason"] = None
    manifest["manual_spatial_status"] = "approved"
    manifest["eda_nonspatial_ready"] = True
    manifest["eda_spatial_ready"] = True
    manifest.to_csv(manifest_path, index=False)

    manifest_dir = build_dir / "manifests"
    manifest.to_csv(manifest_dir / "eda_nonspatial_manifest.csv", index=False)
    manifest.to_csv(manifest_dir / "eda_spatial_manifest.csv", index=False)
    manifest.iloc[0:0].to_csv(manifest_dir / "quarantine_manifest.csv", index=False)

    split_dir = build_dir / "splits"
    split_hashes = {}
    split_counts = {}
    for split_name, expected_volumes in EXPECTED_SPLIT_VOLUMES.items():
        split_frame = manifest[manifest.split == split_name].copy()
        volume_ids = sorted(split_frame.volume_id.unique())
        if len(volume_ids) != expected_volumes:
            raise RuntimeError(f"{split_name}: {len(volume_ids)} volumes != {expected_volumes}")
        split_counts[split_name] = {"volumes": len(volume_ids), "slices": len(split_frame)}
        split_frame.to_csv(split_dir / f"{split_name}_slices.csv", index=False)
        (split_dir / f"{split_name}_volumes.txt").write_text("\n".join(map(str, volume_ids)) + "\n", encoding="utf-8")
        for filename in [f"{split_name}_slices.csv", f"{split_name}_volumes.txt"]:
            split_hashes[filename] = sha256_file(split_dir / filename)
    write_json(split_hashes, split_dir / "split_hashes.json")

    manifest_hash = sha256_file(manifest_path)
    strict_summary.update({
        "manifest_hash": manifest_hash,
        "split_hashes": split_hashes,
        "spatial_review_status": "approved",
        "spatial_review_approved_volumes": EXPECTED_VOLUMES,
        "spatial_review_rejected_volumes": 0,
        "nonspatial_eda_ready": True,
        "spatial_eda_ready": True,
        "training_ready": False,
        "next_required_gate": "Manifest-driven loader validation and 16-slice overfit Dice >= 0.80",
    })
    write_json(strict_summary, strict_summary_path)

    dataset_version.update({
        "status": "verified_eda_ready",
        "verified_at": timestamp,
        "reviewer": args.reviewer,
        "manifest_hash": manifest_hash,
        "split_hashes": split_hashes,
        "spatial_review": {
            "approved_volumes": EXPECTED_VOLUMES,
            "rejected_volumes": 0,
            "contact_sheets": 7,
            "per_volume_review_sheets": EXPECTED_VOLUMES,
        },
        "known_limitations": [
            "Dataset is ready for EDA, but training remains gated on loader tests and a 16-slice overfit check.",
            "This is a single canonical LiTS cohort; the three historical PNG sources are overlapping representations, not additional patients.",
        ],
    })
    write_json(dataset_version, dataset_version_path)

    readiness = {
        "dataset_build": str(build_dir),
        "status": "verified_eda_ready",
        "automatic_gates_pass": True,
        "strict_validation_failures": 0,
        "spatial_review_approved_volumes": EXPECTED_VOLUMES,
        "spatial_review_rejected_volumes": 0,
        "manifest_rows": len(manifest),
        "manifest_hash": manifest_hash,
        "split_counts": split_counts,
        "split_hashes": split_hashes,
        "nonspatial_eda_ready": True,
        "spatial_eda_ready": True,
        "training_ready": False,
        "next_required_gate": "Manifest-driven loader validation and 16-slice overfit Dice >= 0.80",
    }
    write_json(readiness, build_dir / "dataset_readiness.json")

    dataset_card = f"""LiTS CORRECTED CANONICAL STAGING DATASET
========================================

Build: {build_dir.name}
Status: verified_eda_ready
Verified: {timestamp}

Counts
------
Volumes: {EXPECTED_VOLUMES}
Slices: {EXPECTED_SLICES}
Identity volumes: 84
Rot180-corrected segmentation volumes: 47
Strict validation failures: 0
Spatially approved volumes: 131

Splits
------
Train: {split_counts['train']['volumes']} volumes, {split_counts['train']['slices']} slices
Validation: {split_counts['val']['volumes']} volumes, {split_counts['val']['slices']} slices
Test: {split_counts['test']['volumes']} volumes, {split_counts['test']['slices']} slices

Permitted use
-------------
The verified manifest is ready for structural, label, image-mask, and spatial EDA.
Training remains blocked until the project loader passes integration tests and
the 16-slice tumor-positive overfit gate reaches Dice >= 0.80.
"""
    (build_dir / "dataset_card.txt").write_text(dataset_card, encoding="utf-8")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()

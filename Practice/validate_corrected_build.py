"""Strictly validate a corrected LiTS build and create spatial-review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_VOLUMES = 131
EXPECTED_SLICES = 58638
EXPECTED_SIZE = (256, 256)
EXPECTED_SPLIT_VOLUMES = {"train": 104, "val": 13, "test": 14}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def inspect_triplet(args: tuple[int, tuple, Path]) -> dict:
    index, row, build_dir = args
    image_path = build_dir / row.image_path
    organ_path = build_dir / row.organ_mask_path
    tumor_path = build_dir / row.tumor_mask_path
    failures: list[str] = []

    arrays = {}
    modes = {}
    sizes = {}
    for name, path in [("image", image_path), ("organ", organ_path), ("tumor", tumor_path)]:
        if not path.exists():
            failures.append(f"missing_{name}")
            continue
        try:
            with Image.open(path) as opened:
                modes[name] = opened.mode
                sizes[name] = opened.size
                arrays[name] = np.asarray(opened).copy()
        except Exception as error:  # pragma: no cover - diagnostic path
            failures.append(f"unreadable_{name}:{type(error).__name__}")

    if len(arrays) == 3:
        if any(size != EXPECTED_SIZE for size in sizes.values()):
            failures.append("invalid_size")
        if any(mode != "L" for mode in modes.values()):
            failures.append("invalid_mode")

        organ_values = set(np.unique(arrays["organ"]).astype(int).tolist())
        tumor_values = set(np.unique(arrays["tumor"]).astype(int).tolist())
        if not organ_values.issubset({0, 255}):
            failures.append(f"invalid_organ_values:{sorted(organ_values)}")
        if not tumor_values.issubset({0, 255}):
            failures.append(f"invalid_tumor_values:{sorted(tumor_values)}")

        organ = arrays["organ"] > 0
        tumor = arrays["tumor"] > 0
        organ_pixels = int(organ.sum())
        tumor_pixels = int(tumor.sum())
        if np.any(tumor & ~organ):
            failures.append("tumor_outside_organ")
        if organ_pixels != int(row.organ_pixels):
            failures.append("organ_pixel_count_mismatch")
        if tumor_pixels != int(row.tumor_pixels):
            failures.append("tumor_pixel_count_mismatch")
    else:
        organ_pixels = None
        tumor_pixels = None

    return {
        "index": index,
        "sample_id": row.sample_id,
        "volume_id": int(row.volume_id),
        "slice_index": int(row.slice_index),
        "failures": "|".join(failures),
        "passed": not failures,
        "image_min": int(arrays["image"].min()) if "image" in arrays else None,
        "image_max": int(arrays["image"].max()) if "image" in arrays else None,
        "organ_pixels_checked": organ_pixels,
        "tumor_pixels_checked": tumor_pixels,
    }


def plot_overlay(ax, image: np.ndarray, organ: np.ndarray, tumor: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    if np.any(organ):
        ax.contour(organ.astype(float), levels=[0.5], colors=["#22c55e"], linewidths=0.8)
    if np.any(tumor):
        ax.contour(tumor.astype(float), levels=[0.5], colors=["#ef4444"], linewidths=0.8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def load_triplet(build_dir: Path, row) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(build_dir / row.image_path) as opened:
        image = np.asarray(opened).copy()
    with Image.open(build_dir / row.organ_mask_path) as opened:
        organ = np.asarray(opened) > 0
    with Image.open(build_dir / row.tumor_mask_path) as opened:
        tumor = np.asarray(opened) > 0
    return image, organ, tumor


def select_review_rows(group: pd.DataFrame) -> list:
    selected = []
    positive = group[group.tumor_pixels > 0].sort_values("tumor_pixels")
    negative = group[group.tumor_pixels == 0].sort_values("organ_pixels", ascending=False)
    if len(positive):
        selected.append(positive.iloc[-1])
        selected.append(positive.iloc[len(positive) // 2])
    else:
        organ_positive = group[group.organ_pixels > 0].sort_values("organ_pixels")
        if len(organ_positive):
            selected.append(organ_positive.iloc[-1])
            selected.append(organ_positive.iloc[len(organ_positive) // 2])
    if len(negative):
        selected.append(negative.iloc[0])
    while len(selected) < 3:
        selected.append(group.iloc[len(group) // 2])
    return selected[:3]


def generate_spatial_reviews(manifest: pd.DataFrame, build_dir: Path, output_dir: Path) -> pd.DataFrame:
    per_volume_dir = output_dir / "per_volume"
    contact_dir = output_dir / "contact_sheets"
    per_volume_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    representative_rows = []

    for volume_id, group in manifest.groupby("volume_id", sort=True):
        selected = select_review_rows(group)
        representative_rows.append(selected[0])
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
        for axis, row in zip(axes, selected):
            image, organ, tumor = load_triplet(build_dir, row)
            title = f"v{volume_id:03d} s{int(row.slice_index):04d}\norgan={int(row.organ_pixels):,} tumor={int(row.tumor_pixels):,}"
            plot_overlay(axis, image, organ, tumor, title)
        transform = str(group.transform_applied.iloc[0])
        fig.suptitle(f"Spatial review v{volume_id:03d} | transform={transform}", fontsize=11)
        figure_path = per_volume_dir / f"v{volume_id:03d}_review.png"
        fig.tight_layout()
        fig.savefig(figure_path, dpi=145, bbox_inches="tight")
        plt.close(fig)
        review_rows.append({
            "volume_id": int(volume_id),
            "transform_applied": transform,
            "review_figure": str(figure_path),
            "review_status": "pending_manual_review",
            "review_notes": "",
        })

    page_size = 20
    for page, start in enumerate(range(0, len(representative_rows), page_size), start=1):
        page_rows = representative_rows[start : start + page_size]
        columns = 5
        rows = math.ceil(len(page_rows) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(15, rows * 3.0))
        axes = np.atleast_1d(axes).ravel()
        for axis, row in zip(axes, page_rows):
            image, organ, tumor = load_triplet(build_dir, row)
            transform = str(row.transform_applied)
            title = f"v{int(row.volume_id):03d} s{int(row.slice_index):04d} | {transform}"
            plot_overlay(axis, image, organ, tumor, title)
        for axis in axes[len(page_rows) :]:
            axis.axis("off")
        fig.suptitle(f"Corrected LiTS representative spatial review — page {page}", fontsize=13)
        fig.tight_layout()
        fig.savefig(contact_dir / f"contact_sheet_{page:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    review = pd.DataFrame(review_rows)
    review.to_csv(output_dir / "spatial_review.csv", index=False)
    return review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    manifest_path = build_dir / "manifests" / "slice_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    structural_failures = []
    if len(manifest) != EXPECTED_SLICES:
        structural_failures.append(f"rows={len(manifest)} expected={EXPECTED_SLICES}")
    if manifest.volume_id.nunique() != EXPECTED_VOLUMES:
        structural_failures.append(f"volumes={manifest.volume_id.nunique()} expected={EXPECTED_VOLUMES}")
    duplicate_keys = int(manifest.duplicated(["volume_id", "slice_index"]).sum())
    if duplicate_keys:
        structural_failures.append(f"duplicate_keys={duplicate_keys}")
    if manifest[["source_volume_sha256", "source_segmentation_sha256"]].isna().any().any():
        structural_failures.append("missing_source_hashes")
    if not set(manifest.transform_applied.unique()).issubset({"identity", "rot180"}):
        structural_failures.append("unexpected_transform")
    if structural_failures:
        raise RuntimeError("; ".join(structural_failures))

    tasks = [(index, row, build_dir) for index, row in enumerate(manifest.itertuples(index=False))]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for count, result in enumerate(executor.map(inspect_triplet, tasks), start=1):
            results.append(result)
            if count % 5000 == 0 or count == len(tasks):
                print(f"validated {count}/{len(tasks)} slices")

    slice_checks = pd.DataFrame(results).sort_values("index")
    audit_dir = build_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    failures = slice_checks[~slice_checks.passed]
    failures.to_csv(audit_dir / "strict_validation_failures.csv", index=False)

    manifest = manifest.copy()
    manifest["split"] = np.select(
        [manifest.volume_id <= 103, manifest.volume_id <= 116],
        ["train", "val"],
        default="test",
    )
    volume_summary = manifest.groupby(["volume_id", "split", "transform_applied"]).agg(
        slices=("slice_index", "count"),
        organ_positive_slices=("organ_present", "sum"),
        tumor_positive_slices=("tumor_present", "sum"),
        organ_pixels=("organ_pixels", "sum"),
        tumor_pixels=("tumor_pixels", "sum"),
    ).reset_index()
    volume_summary.to_csv(audit_dir / "volume_summary.csv", index=False)

    split_summary = manifest.groupby("split").agg(
        volumes=("volume_id", "nunique"),
        slices=("slice_index", "count"),
        organ_positive_slices=("organ_present", "sum"),
        tumor_positive_slices=("tumor_present", "sum"),
        organ_pixels=("organ_pixels", "sum"),
        tumor_pixels=("tumor_pixels", "sum"),
    ).reset_index()
    split_summary["tumor_positive_rate"] = split_summary.tumor_positive_slices / split_summary.slices
    split_summary.to_csv(audit_dir / "split_summary.csv", index=False)

    split_dir = build_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    split_volume_counts = {}
    for split_name, expected_count in EXPECTED_SPLIT_VOLUMES.items():
        split_frame = manifest[manifest.split == split_name].copy()
        split_frame.to_csv(split_dir / f"{split_name}_slices.csv", index=False)
        volume_ids = sorted(split_frame.volume_id.unique())
        split_volume_counts[split_name] = len(volume_ids)
        (split_dir / f"{split_name}_volumes.txt").write_text("\n".join(map(str, volume_ids)) + "\n", encoding="utf-8")
        if len(volume_ids) != expected_count:
            raise RuntimeError(f"{split_name} volume count {len(volume_ids)} != {expected_count}")
        for name in [f"{split_name}_slices.csv", f"{split_name}_volumes.txt"]:
            split_hashes[name] = sha256_file(split_dir / name)
    write_json(split_hashes, split_dir / "split_hashes.json")

    manifest["verification_status"] = "pending_spatial_review"
    manifest["exclusion_reason"] = None
    manifest.to_csv(manifest_path, index=False)
    manifest[manifest.automatic_integrity_pass].to_csv(build_dir / "manifests" / "eda_nonspatial_manifest.csv", index=False)
    manifest.iloc[0:0].to_csv(build_dir / "manifests" / "eda_spatial_manifest.csv", index=False)
    manifest.iloc[0:0].to_csv(build_dir / "manifests" / "quarantine_manifest.csv", index=False)

    review = generate_spatial_reviews(manifest, build_dir, build_dir / "spatial_reviews")
    all_automatic_pass = failures.empty and not structural_failures
    summary = {
        "build_dir": str(build_dir),
        "manifest_rows": len(manifest),
        "volumes": int(manifest.volume_id.nunique()),
        "derived_png_files_expected": EXPECTED_SLICES * 3,
        "strictly_validated_slices": int(slice_checks.passed.sum()),
        "strict_validation_failures": int(len(failures)),
        "source_hashes_complete": not manifest[["source_volume_sha256", "source_segmentation_sha256"]].isna().any().any(),
        "identity_volumes": int(volume_summary.transform_applied.eq("identity").sum()),
        "rot180_volumes": int(volume_summary.transform_applied.eq("rot180").sum()),
        "split_volume_counts": split_volume_counts,
        "split_hashes": split_hashes,
        "manifest_hash": sha256_file(manifest_path),
        "automatic_gates_pass": bool(all_automatic_pass),
        "spatial_review_volumes": len(review),
        "spatial_review_status": "pending_manual_review",
        "nonspatial_eda_ready": bool(all_automatic_pass),
        "spatial_eda_ready": False,
        "training_ready": False,
    }
    write_json(summary, audit_dir / "strict_validation_summary.json")
    write_json({
        "dataset_name": "LiTS corrected canonical staging",
        "build_id": build_dir.name,
        "status": "automatic_validation_passed_pending_spatial_review" if all_automatic_pass else "failed",
        "counts": {
            "volumes": int(manifest.volume_id.nunique()),
            "slices": len(manifest),
            "organ_positive_slices": int(manifest.organ_present.sum()),
            "tumor_positive_slices": int(manifest.tumor_present.sum()),
            "organ_pixels": int(manifest.organ_pixels.sum()),
            "tumor_pixels": int(manifest.tumor_pixels.sum()),
        },
        "transforms": {
            "identity": sorted(volume_summary.loc[volume_summary.transform_applied == "identity", "volume_id"].astype(int).tolist()),
            "rot180": sorted(volume_summary.loc[volume_summary.transform_applied == "rot180", "volume_id"].astype(int).tolist()),
        },
        "manifest_hash": summary["manifest_hash"],
        "split_hashes": split_hashes,
        "preprocessing_profile": "HU[-160,240]_bilinear_image_nearest_mask_256",
        "known_limitations": ["All 131 spatial review sheets require final approval before spatial EDA or training."],
    }, build_dir / "dataset_version.json")
    print(json.dumps(summary, indent=2))
    if not all_automatic_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

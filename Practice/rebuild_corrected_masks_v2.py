"""Create a corrected LiTS v2 build using resize-first-then-transform labels."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image


EXPECTED_SLICES = 58638
SIZE = (256, 256)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).resize(SIZE, Image.Resampling.NEAREST))


def transform_256(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return mask
    if transform == "rot180":
        return np.rot90(mask, 2).copy()
    raise ValueError(transform)


def save_png_atomic(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary)
    temporary.replace(path)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--reference-build", type=Path, required=True)
    parser.add_argument("--output-build", type=Path, required=True)
    args = parser.parse_args()

    source_build = args.source_build.resolve()
    reference_build = args.reference_build.resolve()
    output_build = args.output_build.resolve()
    if output_build.exists() and any(output_build.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_build}")

    source_manifest = pd.read_csv(source_build / "manifests" / "slice_manifest.csv")
    reference_manifest = pd.read_csv(reference_build / "manifests" / "slice_manifest.csv")
    if len(source_manifest) != EXPECTED_SLICES or len(reference_manifest) != EXPECTED_SLICES:
        raise RuntimeError("Source/reference manifest row count is not canonical")

    reference_counts = reference_manifest.set_index(["volume_id", "slice_index"])[
        ["organ_pixels_256", "tumor_pixels_256", "organ_present_256", "tumor_present_256"]
    ]
    rows = []

    for volume_id, group in source_manifest.groupby("volume_id", sort=True):
        first = group.iloc[0]
        seg_image = nib.load(str(first.source_segmentation_path))
        transform = str(first.transform_applied)
        for row in group.sort_values("slice_index").itertuples(index=False):
            slice_index = int(row.slice_index)
            seg_slice = np.asanyarray(seg_image.dataobj[:, :, slice_index])
            labels = set(np.unique(seg_slice).astype(int).tolist())
            if not labels.issubset({0, 1, 2}):
                raise RuntimeError(f"Invalid labels {labels} at {volume_id}/{slice_index}")

            # Critical order: derive and resize labels first, then apply the
            # evidence-backed in-plane transform on the final 256 grid.
            organ_256 = resize_mask((seg_slice > 0).astype(np.uint8) * 255)
            tumor_256 = resize_mask((seg_slice == 2).astype(np.uint8) * 255)
            organ_256 = transform_256(organ_256, transform)
            tumor_256 = transform_256(tumor_256, transform)
            organ = organ_256 > 0
            tumor = tumor_256 > 0
            if np.any(tumor & ~organ):
                raise RuntimeError(f"Tumor outside organ at {volume_id}/{slice_index}")

            ref = reference_counts.loc[(volume_id, slice_index)]
            if int(organ.sum()) != int(ref.organ_pixels_256):
                raise RuntimeError(f"Organ count drift at {volume_id}/{slice_index}")
            if int(tumor.sum()) != int(ref.tumor_pixels_256):
                raise RuntimeError(f"Tumor count drift at {volume_id}/{slice_index}")

            image_rel = Path(row.image_path)
            organ_rel = Path(row.organ_mask_path)
            tumor_rel = Path(row.tumor_mask_path)
            hardlink_or_copy(source_build / image_rel, output_build / image_rel)
            save_png_atomic(organ_256, output_build / organ_rel)
            save_png_atomic(tumor_256, output_build / tumor_rel)

            record = row._asdict()
            record.update({
                "organ_pixels": int(organ.sum()),
                "tumor_pixels": int(tumor.sum()),
                "organ_present": bool(organ.any()),
                "tumor_present": bool(tumor.any()),
                "automatic_integrity_pass": True,
                "verification_status": "pending_spatial_review",
                "exclusion_reason": None,
                "build_id": output_build.name,
                "mask_operation_order": "derive_resize_nearest_then_transform_256",
            })
            for removable in ["manual_spatial_status", "eda_nonspatial_ready", "eda_spatial_ready", "split"]:
                record.pop(removable, None)
            rows.append(record)

        print(f"rebuilt masks for volume {int(volume_id):03d} ({len(group)} slices, transform={transform})")

    manifest = pd.DataFrame(rows).sort_values(["volume_id", "slice_index"]).reset_index(drop=True)
    if len(manifest) != EXPECTED_SLICES:
        raise RuntimeError(f"Output rows {len(manifest)} != {EXPECTED_SLICES}")
    manifest_dir = output_build / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_dir / "slice_manifest.csv", index=False)
    print(f"created={output_build}")
    print(f"organ_positive={int(manifest.organ_present.sum())}")
    print(f"tumor_positive={int(manifest.tumor_present.sum())}")
    print(f"organ_pixels={int(manifest.organ_pixels.sum())}")
    print(f"tumor_pixels={int(manifest.tumor_pixels.sum())}")


if __name__ == "__main__":
    main()

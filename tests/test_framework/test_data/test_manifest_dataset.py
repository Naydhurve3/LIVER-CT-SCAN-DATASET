import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.framework.data.manifest_dataset import (
    VerifiedManifestDataset,
    manifest_sample_weights,
    create_manifest_dataloaders,
)


FIELDNAMES = [
    "sample_id",
    "volume_id",
    "slice_index",
    "image_path",
    "organ_mask_path",
    "tumor_mask_path",
    "image_width",
    "image_height",
    "organ_pixels",
    "tumor_pixels",
    "automatic_integrity_pass",
    "verification_status",
    "exclusion_reason",
    "split",
    "manual_spatial_status",
]


def _build_manifest(tmp_path: Path) -> Path:
    rows = []
    for split, volume_id, positive in [
        ("train", 0, False),
        ("train", 0, True),
        ("val", 1, True),
        ("test", 2, True),
    ]:
        sample_id = f"v{volume_id:03d}_s{len(rows):04d}"
        image_rel = Path("images") / f"{sample_id}.png"
        organ_rel = Path("organ_masks") / f"{sample_id}.png"
        tumor_rel = Path("tumor_masks") / f"{sample_id}.png"
        for rel in (image_rel, organ_rel, tumor_rel):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)

        image = np.arange(64, dtype=np.uint8).reshape(8, 8)
        organ = np.zeros((8, 8), dtype=np.uint8)
        organ[1:7, 1:7] = 255
        tumor = np.zeros((8, 8), dtype=np.uint8)
        if positive:
            tumor[3:5, 3:5] = 255
        Image.fromarray(image).save(tmp_path / image_rel)
        Image.fromarray(organ).save(tmp_path / organ_rel)
        Image.fromarray(tumor).save(tmp_path / tumor_rel)

        rows.append(
            {
                "sample_id": sample_id,
                "volume_id": volume_id,
                "slice_index": len(rows),
                "image_path": str(image_rel),
                "organ_mask_path": str(organ_rel),
                "tumor_mask_path": str(tumor_rel),
                "image_width": 8,
                "image_height": 8,
                "organ_pixels": 36,
                "tumor_pixels": 4 if positive else 0,
                "automatic_integrity_pass": True,
                "verification_status": "verified",
                "exclusion_reason": "",
                "split": split,
                "manual_spatial_status": "approved",
            }
        )

    manifest = tmp_path / "manifests" / "slice_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_manifest_dataset_loads_train_rows_and_metadata(tmp_path):
    manifest = _build_manifest(tmp_path)
    dataset = VerifiedManifestDataset(
        manifest, "train", root_dir=tmp_path, target="tumor"
    )
    assert len(dataset) == 2
    item = dataset[1]
    assert item["image"].shape == (1, 8, 8)
    assert item["mask"].shape == (1, 8, 8)
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}
    assert item["sample_id"] == "v000_s0001"
    assert item["split"] == "train"


def test_manifest_dataset_locks_test_split_by_default(tmp_path):
    manifest = _build_manifest(tmp_path)
    with pytest.raises(PermissionError, match="Test data access is locked"):
        VerifiedManifestDataset(manifest, "test", root_dir=tmp_path)


def test_manifest_dataset_allows_explicit_one_time_test_access(tmp_path):
    manifest = _build_manifest(tmp_path)
    dataset = VerifiedManifestDataset(
        manifest, "test", root_dir=tmp_path, allow_test=True
    )
    assert len(dataset) == 1


def test_manifest_dataset_preserves_requested_sample_order(tmp_path):
    manifest = _build_manifest(tmp_path)
    requested = ["v000_s0001", "v000_s0000"]
    dataset = VerifiedManifestDataset(
        manifest, "train", root_dir=tmp_path, sample_ids=requested
    )
    assert dataset.sample_ids == requested


def test_manifest_dataset_rejects_pixel_count_mismatch(tmp_path):
    manifest = _build_manifest(tmp_path)
    dataset = VerifiedManifestDataset(manifest, "train", root_dir=tmp_path)
    dataset.rows[1]["tumor_pixels"] = 99
    with pytest.raises(ValueError, match="pixel count mismatch"):
        _ = dataset[1]


def test_manifest_sample_weights_use_manifest_tumor_presence(tmp_path):
    manifest = _build_manifest(tmp_path)
    dataset = VerifiedManifestDataset(manifest, "train", root_dir=tmp_path)
    assert manifest_sample_weights(dataset, 3.0).tolist() == [1.0, 3.0]


def test_create_manifest_dataloaders_exposes_image_and_mask(tmp_path):
    manifest = _build_manifest(tmp_path)
    train_loader, val_loader, test_loader = create_manifest_dataloaders(
        manifest, root_dir=tmp_path, batch_size=2, tumor_sampler_weight=None
    )
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert test_loader is None
    batch = next(iter(train_loader))
    assert batch["image"].shape[1] == 1 and batch["image"].shape[2] == 8
    assert batch["mask"].shape == batch["image"].shape
    assert "sample_id" in batch


def test_create_manifest_dataloaders_opt_in_test(tmp_path):
    manifest = _build_manifest(tmp_path)
    train_loader, val_loader, test_loader = create_manifest_dataloaders(
        manifest, root_dir=tmp_path, allow_test=True, tumor_sampler_weight=None
    )
    assert test_loader is not None
    assert len(test_loader.dataset) == 1


def test_create_manifest_dataloaders_limit_slices(tmp_path):
    manifest = _build_manifest(tmp_path)
    train_loader, val_loader, _ = create_manifest_dataloaders(
        manifest, root_dir=tmp_path, limit_slices=1, tumor_sampler_weight=None
    )
    assert len(train_loader.dataset) == 1
    assert len(val_loader.dataset) == 1

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REQUIRED_COLUMNS = {
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
}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _relative_path(root: Path, value: str) -> Path:
    normalized = str(value).replace("\\", os.sep).replace("/", os.sep)
    path = Path(normalized)
    return path if path.is_absolute() else root / path


class VerifiedManifestDataset(Dataset):
    """Strict 2D LiTS loader backed by the verified slice manifest.

    Test access is deliberately opt-in. Training and validation notebooks should
    leave ``allow_test=False`` so an accidental split change fails immediately.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        root_dir: str | Path | None = None,
        target: str = "tumor",
        transform=None,
        sample_ids: Optional[Sequence[str]] = None,
        require_verified: bool = True,
        allow_test: bool = False,
        validate_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        split = str(split).strip().lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")
        if split == "test" and not allow_test:
            raise PermissionError(
                "Test data access is locked. Pass allow_test=True only for the "
                "one-time held-out evaluation after model decisions are frozen."
            )
        if target not in {"tumor", "organ"}:
            raise ValueError("target must be either 'tumor' or 'organ'")

        self.split = split
        self.target = target
        self.transform = transform
        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else self.manifest_path.parent.parent
        )

        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(
                    "Manifest is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            rows = [dict(row) for row in reader if row.get("split", "").lower() == split]

        if require_verified:
            rows = [
                row
                for row in rows
                if row.get("verification_status", "").lower() == "verified"
                and _as_bool(row.get("automatic_integrity_pass"))
                and row.get("manual_spatial_status", "").lower() == "approved"
                and not row.get("exclusion_reason", "").strip()
            ]

        if sample_ids is not None:
            requested = list(sample_ids)
            if len(requested) != len(set(requested)):
                raise ValueError("sample_ids contains duplicates")
            by_id = {row["sample_id"]: row for row in rows}
            missing_ids = [sample_id for sample_id in requested if sample_id not in by_id]
            if missing_ids:
                preview = ", ".join(missing_ids[:10])
                raise KeyError(
                    f"{len(missing_ids)} requested sample IDs are unavailable in "
                    f"split={split}: {preview}"
                )
            rows = [by_id[sample_id] for sample_id in requested]

        if not rows:
            raise ValueError(f"No eligible rows found for split={split}")
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs found in split={split}")

        for row in rows:
            row["volume_id"] = int(row["volume_id"])
            row["slice_index"] = int(row["slice_index"])
            row["image_width"] = int(row["image_width"])
            row["image_height"] = int(row["image_height"])
            row["organ_pixels"] = int(row["organ_pixels"])
            row["tumor_pixels"] = int(row["tumor_pixels"])
            row["image_path"] = _relative_path(self.root_dir, row["image_path"])
            row["organ_mask_path"] = _relative_path(
                self.root_dir, row["organ_mask_path"]
            )
            row["tumor_mask_path"] = _relative_path(
                self.root_dir, row["tumor_mask_path"]
            )

        if validate_paths:
            missing_paths: List[Path] = []
            mask_key = f"{target}_mask_path"
            for row in rows:
                for path in (row["image_path"], row[mask_key]):
                    if not path.is_file():
                        missing_paths.append(path)
                        if len(missing_paths) >= 10:
                            break
                if len(missing_paths) >= 10:
                    break
            if missing_paths:
                rendered = "\n".join(f"- {path}" for path in missing_paths)
                raise FileNotFoundError(f"Manifest paths are missing:\n{rendered}")

        self.rows: List[Dict] = rows

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def sample_ids(self) -> List[str]:
        return [row["sample_id"] for row in self.rows]

    @property
    def tumor_positive_flags(self) -> List[bool]:
        return [row["tumor_pixels"] > 0 for row in self.rows]

    def __getitem__(self, index: int) -> Dict:
        row = self.rows[index]
        mask_path = row[f"{self.target}_mask_path"]

        with Image.open(row["image_path"]) as image_file:
            image = np.asarray(image_file.convert("L"), dtype=np.float32) / 255.0
        with Image.open(mask_path) as mask_file:
            mask = (np.asarray(mask_file.convert("L"), dtype=np.uint8) > 0).astype(
                np.float32
            )

        expected_shape = (row["image_height"], row["image_width"])
        if image.shape != expected_shape:
            raise ValueError(
                f"{row['sample_id']} image shape {image.shape} != {expected_shape}"
            )
        if mask.shape != expected_shape:
            raise ValueError(
                f"{row['sample_id']} mask shape {mask.shape} != {expected_shape}"
            )

        expected_pixels = row[f"{self.target}_pixels"]
        observed_pixels = int(mask.sum())
        if observed_pixels != expected_pixels:
            raise ValueError(
                f"{row['sample_id']} {self.target} pixel count mismatch: "
                f"manifest={expected_pixels}, file={observed_pixels}"
            )

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "sample_id": row["sample_id"],
            "volume_id": row["volume_id"],
            "slice_index": row["slice_index"],
            "tumor_pixels": row["tumor_pixels"],
            "split": row["split"],
        }


def manifest_sample_weights(
    dataset: VerifiedManifestDataset, tumor_positive_weight: float = 3.0
) -> torch.DoubleTensor:
    if tumor_positive_weight <= 0:
        raise ValueError("tumor_positive_weight must be positive")
    weights: Iterable[float] = (
        tumor_positive_weight if flag else 1.0
        for flag in dataset.tumor_positive_flags
    )
    return torch.as_tensor(list(weights), dtype=torch.double)


def create_manifest_dataloaders(
    manifest_path: str | Path,
    *,
    root_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = True,
    transform_train=None,
    transform_val=None,
    tumor_sampler_weight: float | None = 3.0,
    seed: int = 42,
    limit_slices: int | None = None,
    allow_test: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Build train/val (and locked test) DataLoaders from the verified manifest.

    Every batch exposes ``image`` (1xHxW) and ``mask`` (1xHxW) tensors plus
    ``sample_id``/``volume_id`` metadata, matching ``ResearchTrainer`` and the
    legacy ``Trainer``. Test access is opt-in via ``allow_test``; otherwise the
    third loader is ``None`` and any accidental test split change fails fast
    inside ``VerifiedManifestDataset``.
    """
    train_dataset = VerifiedManifestDataset(
        manifest_path, "train", root_dir=root_dir, transform=transform_train
    )
    val_dataset = VerifiedManifestDataset(
        manifest_path, "val", root_dir=root_dir, transform=transform_val
    )
    if allow_test:
        test_dataset = VerifiedManifestDataset(
            manifest_path, "test", root_dir=root_dir,
            transform=transform_val, allow_test=True,
        )
    else:
        test_dataset = None

    if limit_slices is not None:
        if limit_slices <= 0:
            raise ValueError("limit_slices must be positive")
        train_dataset = VerifiedManifestDataset(
            manifest_path, "train", root_dir=root_dir, transform=transform_train,
            sample_ids=train_dataset.sample_ids[:limit_slices],
        )
        val_dataset = VerifiedManifestDataset(
            manifest_path, "val", root_dir=root_dir, transform=transform_val,
            sample_ids=val_dataset.sample_ids[:limit_slices],
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    if tumor_sampler_weight is not None:
        weights = manifest_sample_weights(train_dataset, tumor_sampler_weight)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
    return train_loader, val_loader, test_loader

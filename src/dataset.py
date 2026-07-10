"""
Dataset classes for the LiTS 2017 dataset with GPU support, error handling,
metadata extraction, and memory-efficient loading.
"""
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DEVICE, TRAIN_CONFIG, NORMALIZATION, NUM_CLASSES,
    DATA_DIR, SPLITS_DIR, STATISTICS_FILE,
)
from src.utils import (
    logger, check_file_exists, load_volume_safe, load_split_ids,
    load_json, numpy_to_gpu, to_device,
)


class LiTSDataset(Dataset):
    """
    PyTorch Dataset for the LiTS (Liver Tumor Segmentation) 2017 dataset.

    Loads 3D CT volumes and segmentation masks. Supports:
    - GPU tensor conversion
    - Metadata extraction (spacing, affine, shape)
    - Memory-mapped loading for large volumes
    - Z-score normalization with clipping
    """

    def __init__(
        self,
        volume_ids: List[str],
        volume_dir: Optional[Path] = None,
        mask_dir: Optional[Path] = None,
        transform=None,
        normalize: bool = True,
        clip_range: Tuple[float, float] = (NORMALIZATION["clip_min"], NORMALIZATION["clip_max"]),
        preload: bool = False,
        memory_map: bool = True,
    ):
        """
        Args:
            volume_ids: List of volume identifiers (e.g., "volume-0").
            volume_dir: Directory containing CT volumes. Defaults to DATA_DIR / "volumes".
            mask_dir: Directory containing segmentation masks. Defaults to DATA_DIR / "masks".
            transform: Optional callable for data augmentation.
            normalize: Whether to apply Z-score normalization.
            clip_range: HU intensity clipping range.
            preload: If True, load all data into RAM at init (fast but memory-heavy).
            memory_map: If True, use NIfTI memory mapping for lazy loading.
        """
        self.volume_ids = volume_ids
        self.volume_dir = volume_dir or (DATA_DIR / "volumes")
        self.mask_dir = mask_dir or (DATA_DIR / "masks")
        self.transform = transform
        self.normalize = normalize
        self.clip_range = clip_range
        self.preload = preload
        self.memory_map = memory_map

        # Cache for preloaded data
        self._volumes: Dict[str, np.ndarray] = {}
        self._masks: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, dict] = {}

        # Load metadata / statistics
        self.stats = load_json(STATISTICS_FILE)

        # If preloading, load everything at init
        if self.preload:
            self._preload_all()

        logger.info(f"LiTSDataset initialized with {len(volume_ids)} volumes")

    def _resolve_path(self, vid: str, suffix: str = ".nii") -> Path:
        """Find the actual file path (handles .nii / .nii.gz)."""
        base = self.volume_dir / vid
        if suffix == ".nii.gz":
            base = self.volume_dir / (vid + ".nii.gz")
            if base.exists():
                return base
        for ext in [".nii", ".nii.gz"]:
            p = self.volume_dir / (vid + ext)
            if p.exists():
                return p
        return base

    def _preload_all(self) -> None:
        """Load all volumes and masks into RAM."""
        for vid in self.volume_ids:
            vol = self._load_volume(vid)
            mask = self._load_mask(vid)
            if vol is not None:
                self._volumes[vid] = vol
            if mask is not None:
                self._masks[vid] = mask
        logger.info(f"Preloaded {len(self._volumes)} volumes, {len(self._masks)} masks")

    def _load_volume(self, vid: str) -> Optional[np.ndarray]:
        """Load a single CT volume with memory mapping support."""
        try:
            vol_path = self._resolve_path(vid)
            if not vol_path.exists():
                logger.warning(f"Volume not found: {vol_path}")
                return None

            img = nib.load(str(vol_path))
            data = img.get_fdata(dtype=np.float32)

            # Extract metadata
            self._metadata[vid] = {
                "shape": data.shape,
                "spacing": tuple(img.header.get_zooms()[:3]) if len(img.header.get_zooms()) >= 3 else (1.0, 1.0, 1.0),
                "affine": img.affine.tolist(),
                "orientation": nib.orientations.aff2axcodes(img.affine),
            }

            return data.astype(np.float32)
        except Exception as e:
            logger.error(f"❌ Error loading volume {vid}: {e}")
            return None

    def _load_mask(self, vid: str) -> Optional[np.ndarray]:
        """Load a single segmentation mask."""
        try:
            mask_path = self._resolve_path(vid.replace("volume", "segmentation"))
            if not mask_path.exists():
                # Try with mask prefix
                mask_path = self.mask_dir / f"{vid}.nii"
                if not mask_path.exists():
                    mask_path = self.mask_dir / f"{vid}.nii.gz"

            if not mask_path.exists():
                logger.warning(f"Mask not found for {vid}")
                return None

            img = nib.load(str(mask_path))
            data = img.get_fdata(dtype=np.int16)
            return data.astype(np.int16)
        except Exception as e:
            logger.error(f"❌ Error loading mask for {vid}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.volume_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a volume-mask pair, optionally transformed and on GPU."""
        vid = self.volume_ids[idx]

        # Get volume — from cache or disk
        if vid in self._volumes:
            volume = self._volumes[vid].copy()
        else:
            vol = self._load_volume(vid)
            if vol is None:
                # Return dummy data if loading failed
                volume = np.zeros((1, 64, 64, 64), dtype=np.float32)
            else:
                volume = vol

        # Get mask — from cache or disk
        if vid in self._masks:
            mask = self._masks[vid].copy()
        else:
            msk = self._load_mask(vid)
            if msk is None:
                mask = np.zeros_like(volume, dtype=np.int16)
            else:
                mask = msk

        # Normalize volume
        if self.normalize:
            volume = self._normalize(volume)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume = np.expand_dims(volume, axis=0).astype(np.float32)
        mask = mask.astype(np.int64)

        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).float()
        mask_tensor = torch.from_numpy(mask).long()

        # Apply transforms (augmentation)
        if self.transform:
            augmented = self.transform(volume_tensor, mask_tensor)
            volume_tensor = augmented["image"]
            mask_tensor = augmented["mask"]

        return {
            "image": to_device(volume_tensor),
            "mask": to_device(mask_tensor),
            "volume_id": vid,
        }

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        """Apply clipping and Z-score normalization."""
        # Clip to HU range
        volume = np.clip(volume, self.clip_range[0], self.clip_range[1])

        # Z-score normalization
        mean = volume.mean()
        std = volume.std()
        if std > 1e-8:
            volume = (volume - mean) / std
        else:
            volume = volume - mean

        return volume

    def get_metadata(self, vid: str) -> dict:
        """Return metadata for a given volume ID."""
        if vid in self._metadata:
            return self._metadata[vid]
        # Trigger load to get metadata
        self._load_volume(vid)
        return self._metadata.get(vid, {})

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights from the dataset to handle imbalance."""
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        total_pixels = 0

        for vid in self.volume_ids:
            mask = self._load_mask(vid)
            if mask is not None:
                for c in range(NUM_CLASSES):
                    class_counts[c] += (mask == c).sum()
                total_pixels += mask.size

        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = total_pixels / (NUM_CLASSES * class_counts)
        weights = weights / weights.sum()  # Normalize
        return torch.from_numpy(weights).float().to(DEVICE)


# =============================================================================
# 2. DATALOADER FACTORY
# =============================================================================
def create_dataloaders(
    batch_size: int = TRAIN_CONFIG["batch_size"],
    num_workers: int = TRAIN_CONFIG["num_workers"],
    pin_memory: bool = TRAIN_CONFIG["pin_memory"],
    transform_train=None,
    transform_val=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from split files."""
    train_ids = load_split_ids(SPLITS_DIR / "train_volumes.txt")
    val_ids = load_split_ids(SPLITS_DIR / "val_volumes.txt")
    test_ids = load_split_ids(SPLITS_DIR / "test_volumes.txt")

    train_dataset = LiTSDataset(train_ids, transform=transform_train)
    val_dataset = LiTSDataset(val_ids, transform=transform_val)
    test_dataset = LiTSDataset(test_ids, transform=transform_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"DataLoaders created — Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    print(f"  DEVICE: {DEVICE}")
    print(f"  LiTSDataset, create_dataloaders available")
    print("  OK (requires .nii files for full test)")

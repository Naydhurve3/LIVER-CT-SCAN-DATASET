"""
Data loading utilities for 2D PNG liver tumor segmentation.
Provides path management, volume indexing, and PyTorch datasets.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image


# ============================================================================
# CONFIGURATION
# ============================================================================
class DatasetConfig:
    """Centralized configuration for dataset paths and settings.

    All paths are derived dynamically from this file's location,
    making the project portable across machines.
    """

    # Root: resolves to <project_root>/src/data_loader.py -> <project_root>
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR = PROJECT_DIR.parent.parent  # e.g., D:\DATA SCIENCE AND ANALYTICS
    DATASET_DIR = BASE_DIR / "Dataset"

    # Data directories
    IMAGES_DIR = DATASET_DIR / "Liver Img Dataset"
    MASKS_DIR = DATASET_DIR / "LiTS_masks"

    # Output directories
    OUTPUT_DIR = PROJECT_DIR / "data"
    SPLITS_DIR = OUTPUT_DIR / "splits"
    METADATA_DIR = OUTPUT_DIR / "metadata"

    # Data parameters
    IMAGE_SIZE = (256, 256)
    SPLIT_RATIOS = (0.8, 0.1, 0.1)

    # Output sub-directories (figures, reports, generated configs)
    EDA_OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda"
    EDA_PLOTS_DIR = EDA_OUTPUT_DIR / "plots"
    PREP_OUTPUT_DIR = PROJECT_DIR / "outputs" / "preprocessing"


# ============================================================================
# DATA PATH MANAGER
# ============================================================================
class DataPathManager:
    """Scans directories, extracts volume IDs, groups slices by volume.

    Works with two naming conventions:
    - Images: Volume-{vol:03d}-{slice:03d}.png
    - Masks: mask-{vol:03d}-{slice:03d}.png
    """

    def __init__(self):
        self.image_dir = DatasetConfig.IMAGES_DIR
        self.mask_dir = DatasetConfig.MASKS_DIR
        self.image_paths: Dict[int, List[Path]] = defaultdict(list)
        self.mask_paths: Dict[int, List[Path]] = defaultdict(list)

    def extract_volume_id(self, filename: str) -> Optional[int]:
        """Extract volume ID from filename pattern.

        Handles: Volume-XXX-YYY.png and mask-XXX-YYY.png
        Returns None if pattern doesn't match.
        """
        match = re.match(r"(?:Volume|mask)-(\d+)-\d+\.png", filename)
        return int(match.group(1)) if match else None

    def extract_slice_id(self, filename: str) -> int:
        """Extract slice number from filename."""
        match = re.search(r"-(\d+)\.png$", str(filename))
        return int(match.group(1)) if match else 0

    def build_index(self) -> Dict:
        """Scan directories and build complete volume index.

        Returns dict with:
          - 'image_paths': {vol_id: [sorted slice paths]}
          - 'mask_paths': {vol_id: [sorted slice paths]}
          - 'volumes': sorted list of all volume IDs
        """
        volume_index = {
            'image_paths': {},
            'mask_paths': {},
            'volumes': []
        }

        # Scan image files
        print("[INFO] Scanning image directory...")
        for img_file in sorted(self.image_dir.glob("Volume-*.png")):
            vid = self.extract_volume_id(img_file.name)
            if vid is not None:
                self.image_paths[vid].append(img_file)

        # Scan mask files
        print("[INFO] Scanning mask directory...")
        for mask_file in sorted(self.mask_dir.glob("mask-*.png")):
            vid = self.extract_volume_id(mask_file.name)
            if vid is not None:
                self.mask_paths[vid].append(mask_file)

        # Sort slices within each volume by slice number
        for vid in self.image_paths:
            self.image_paths[vid] = sorted(
                self.image_paths[vid],
                key=lambda p: self.extract_slice_id(p.name)
            )
        for vid in self.mask_paths:
            self.mask_paths[vid] = sorted(
                self.mask_paths[vid],
                key=lambda p: self.extract_slice_id(p.name)
            )

        volume_index['image_paths'] = dict(self.image_paths)
        volume_index['mask_paths'] = dict(self.mask_paths)
        volume_index['volumes'] = sorted(self.image_paths.keys())

        print(f"[INFO] Found {len(self.image_paths)} image volumes, "
              f"{len(self.mask_paths)} mask volumes")

        return volume_index


# ============================================================================
# VOLUME-WISE SPLITTER
# ============================================================================
class VolumeWiseSplitter:
    """Split volumes into train/val/test (volume-wise, NOT slice-wise).

    This prevents data leakage by ensuring all slices from one volume
    belong to only one split. Split files use comma-separated format:
    e.g., "0,1,2,3,4,5,...,103"
    """

    def __init__(self, split_ratios=(0.8, 0.1, 0.1)):
        self.split_ratios = split_ratios
        self.train_ratio = split_ratios[0]
        self.val_ratio = split_ratios[0] + split_ratios[1]

    def split(self, volume_ids: List[int]) -> Dict[str, List[int]]:
        """Split volume IDs into train/val/test based on ratios.

        Args:
            volume_ids: Sorted list of volume IDs.
        Returns:
            {'train': [...], 'val': [...], 'test': [...]}
        """
        sorted_ids = sorted(volume_ids)
        n = len(sorted_ids)

        train_end = int(n * self.train_ratio)
        val_end = int(n * self.val_ratio)

        splits = {
            'train': sorted_ids[:train_end],
            'val': sorted_ids[train_end:val_end],
            'test': sorted_ids[val_end:]
        }

        print(f"[INFO] Split: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")
        return splits

    def load_splits(self, splits_dir: Path) -> Dict[str, List[int]]:
        """Load volume splits from comma-separated text files.

        Handles the existing format where IDs are comma-separated
        on a single line: "0,1,2,3,...,103"

        Args:
            splits_dir: Directory containing train/val/test_volumes.txt
        Returns:
            {'train': [...], 'val': [...], 'test': [...]}
        """
        splits = {}
        for split_name in ['train', 'val', 'test']:
            filepath = splits_dir / f"{split_name}_volumes.txt"
            if not filepath.exists():
                print(f"[WARNING] {filepath} not found")
                splits[split_name] = []
                continue

            with open(filepath, 'r') as f:
                content = f.read().strip()

            if ',' in content:
                # Comma-separated format: "0,1,2,3,...,103"
                splits[split_name] = [int(x.strip()) for x in content.split(',') if x.strip()]
            else:
                # Newline-separated format (for future compatibility)
                splits[split_name] = [int(line.strip()) for line in content.split('\n') if line.strip()]

            print(f"[INFO] Loaded {split_name}: {len(splits[split_name])} volumes")

        return splits

    def save_splits(self, splits: Dict[str, List[int]], output_dir: Path):
        """Save volume splits to separate files (newline-separated).

        Does NOT overwrite existing files - only creates new ones
        in a separate directory if needed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, vol_ids in splits.items():
            filepath = output_dir / f"{split_name}_volumes.txt"
            with open(filepath, 'w') as f:
                for vid in sorted(vol_ids):
                    f.write(f"{vid}\n")
            print(f"[INFO] Saved {split_name}: {len(vol_ids)} volumes to {filepath}")


# ============================================================================
# 2D PNG DATASET
# ============================================================================
class LiverTumor2DDataset(Dataset):
    """PyTorch Dataset for 2D PNG slices from liver CT + tumor masks.

    Loads single 2D slices from volume-organized PNG files.
    Returns dict with 'image', 'mask', 'volume_id', 'slice_id'.

    Args:
        volume_ids: List of volume IDs to include.
        volume_index: Dict from DataPathManager.build_index().
        transform: Optional callable (img, mask) -> (img, mask).
    """

    def __init__(self, volume_ids: List[int], volume_index: Dict,
                 transform=None, precompute_tumor_flags=True):
        self.volume_ids = volume_ids
        self.volume_index = volume_index
        self.transform = transform

        # Build flat list of (volume_id, slice_idx) tuples
        self.samples = []
        for vid in volume_ids:
            vid_int = int(vid) if not isinstance(vid, int) else vid
            if vid_int in volume_index['image_paths']:
                n_slices = len(volume_index['image_paths'][vid_int])
                for sid in range(n_slices):
                    self.samples.append((vid_int, sid))

        print(f"[INFO] Dataset: {len(self.samples):,} slices from {len(volume_ids)} volumes")

        # Precompute tumor-flags: which samples have any tumor pixel
        self.tumor_flags = None
        if precompute_tumor_flags:
            print("[INFO] Pre-computing tumor flags (scanning masks)...")
            flags = []
            for vid, sid in self.samples:
                has_tumor = False
                if vid in volume_index['mask_paths']:
                    mask_sid = min(sid, len(volume_index['mask_paths'][vid]) - 1)
                    mask_path = volume_index['mask_paths'][vid][mask_sid]
                    try:
                        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
                        has_tumor = bool((mask > 0.5).any())
                    except Exception:
                        pass
                flags.append(has_tumor)
            self.tumor_flags = np.array(flags, dtype=bool)
            n_tumor = int(self.tumor_flags.sum())
            pct = 100.0 * n_tumor / max(len(flags), 1)
            print(f"[INFO]   Tumor slices: {n_tumor}/{len(flags)} ({pct:.2f}%)")
            # Estimate batch coverage (pct of batches with >=1 tumor slice at batch_size=4)
            p_tumor = n_tumor / max(len(flags), 1)
            coverage = 100.0 * (1.0 - (1.0 - p_tumor) ** 4)
            print(f"[INFO]   Est. batches with tumor (bs=4): {coverage:.1f}%")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        vid, sid = self.samples[idx]

        # Load image (grayscale, normalize to [0, 1])
        img_path = self.volume_index['image_paths'][vid][sid]
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0

        # Load mask (values are 0 or 1 in PNG; no /255 needed)
        if vid in self.volume_index['mask_paths']:
            mask_sid = min(sid, len(self.volume_index['mask_paths'][vid]) - 1)
            mask_path = self.volume_index['mask_paths'][vid][mask_sid]
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.float32)
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros_like(img)

        # Apply transforms
        if self.transform:
            img, mask = self.transform(img, mask)

        # Add channel dimension: (H, W) -> (1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'volume_id': vid,
            'slice_id': sid,
            'has_tumor': bool(self.tumor_flags[idx]) if self.tumor_flags is not None else False,
        }


# ============================================================================
# DATALOADER FACTORY
# ============================================================================
def make_tumor_sampler(dataset: LiverTumor2DDataset, batch_size: int = 8,
                       tumor_weight: float = 10.0) -> Optional[WeightedRandomSampler]:
    """Create a WeightedRandomSampler that oversamples tumor-positive slices.

    Args:
        dataset: Dataset with precomputed tumor_flags.
        batch_size: Desired batch size.
        tumor_weight: Multiplier for tumor slice sampling weight.
                      tumor slices get weight = tumor_weight, bg slices get weight = 1.

    Returns:
        WeightedRandomSampler or None if no tumor flags available.
    """
    if dataset.tumor_flags is None:
        print("[WARN] No tumor flags — skipping stratified sampler.")
        return None
    n = len(dataset)
    weights = np.where(dataset.tumor_flags, float(tumor_weight), 1.0)
    print(f"[INFO] Tumor sampler: weight={tumor_weight} for tumor, 1 for bg")
    print(f"[INFO]   Tumor idx weight: {weights[dataset.tumor_flags].mean():.2f} avg, "
          f"bg idx weight: {weights[~dataset.tumor_flags].mean():.2f} avg")
    num_samples = n  # one epoch's worth
    return WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples, replacement=True)


def create_2d_dataloaders(
    volume_index: Dict,
    train_vids: List[int],
    val_vids: List[int],
    test_vids: List[int],
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform_train=None,
    transform_val=None,
    use_tumor_sampler: bool = False,
    tumor_sampler_weight: float = 10.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for 2D PNG data.

    Args:
        volume_index: From DataPathManager.build_index().
        train_vids/val_vids/test_vids: Volume ID lists.
        batch_size: Batch size (default 8 fits RTX 3050 Ti 4GB).
        num_workers: Parallel data loading workers.
        pin_memory: Enable for GPU training.
        transform_train: Optional training augmentation.
        transform_val: Optional validation transform.
        use_tumor_sampler: If True, use WeightedRandomSampler oversampling tumor slices.
        tumor_sampler_weight: Weight multiplier for tumor-positive slices.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = LiverTumor2DDataset(train_vids, volume_index, transform=transform_train)
    val_ds = LiverTumor2DDataset(val_vids, volume_index, transform=transform_val)
    test_ds = LiverTumor2DDataset(test_vids, volume_index, transform=transform_val)

    # Build train loader (optionally with stratified sampler)
    if use_tumor_sampler:
        sampler = make_tumor_sampler(train_ds, batch_size, tumor_weight=tumor_sampler_weight)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0) and (len(train_ds) > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=(len(train_ds) > 0),
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=(num_workers > 0) and (len(train_ds) > 0),
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0) and (len(val_ds) > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0) and (len(test_ds) > 0),
    )

    print(f"[INFO] DataLoaders created:")
    print(f"   Train: {len(train_loader.dataset):,} slices ({len(train_vids)} volumes)")
    if use_tumor_sampler:
        print(f"   Train sampler: stratified (tumor_weight={tumor_sampler_weight})")
    print(f"   Val:   {len(val_loader.dataset):,} slices ({len(val_vids)} volumes)")
    print(f"   Test:  {len(test_loader.dataset):,} slices ({len(test_vids)} volumes)")

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    print(f"  DATASET_DIR: {DatasetConfig.DATASET_DIR}")
    mgr = DataPathManager()
    idx = mgr.build_index()
    print(f"  Volumes: {len(idx['volumes'])}")
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(DatasetConfig.SPLITS_DIR)
    print(f"  Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    ds = LiverTumor2DDataset([0, 1], idx)
    print(f"  Dataset from 2 volumes: {len(ds)} slices")
    print("  OK")

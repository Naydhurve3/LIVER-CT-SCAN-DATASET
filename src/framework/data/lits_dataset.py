import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.framework.core.registry import DATASETS


class DataPathManager:
    def __init__(self, images_dir, masks_dir, strict: bool = False):
        self.image_dir = Path(images_dir)
        self.mask_dir = Path(masks_dir)
        self.image_paths: Dict[int, List[Path]] = defaultdict(list)
        self.mask_paths: Dict[int, List[Path]] = defaultdict(list)
        self.strict = strict

    def extract_volume_id(self, filename: str) -> Optional[int]:
        match = re.match(r"(?:Volume|mask)-(\d+)-\d+\.png", filename)
        return int(match.group(1)) if match else None

    def extract_slice_id(self, filename: str) -> int:
        match = re.search(r"-(\d+)\.png$", str(filename))
        return int(match.group(1)) if match else 0

    def build_index(self) -> Dict:
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        volume_index = {'image_paths': {}, 'mask_paths': {}, 'volumes': []}
        for img_file in sorted(self.image_dir.glob("Volume-*.png")):
            vid = self.extract_volume_id(img_file.name)
            if vid is not None:
                self.image_paths[vid].append(img_file)
        for mask_file in sorted(self.mask_dir.glob("mask-*.png")):
            vid = self.extract_volume_id(mask_file.name)
            if vid is not None:
                self.mask_paths[vid].append(mask_file)
        for vid in self.image_paths:
            self.image_paths[vid] = sorted(
                self.image_paths[vid], key=lambda p: self.extract_slice_id(p.name)
            )
        for vid in self.mask_paths:
            self.mask_paths[vid] = sorted(
                self.mask_paths[vid], key=lambda p: self.extract_slice_id(p.name)
            )
        volume_index['image_paths'] = dict(self.image_paths)
        volume_index['mask_paths'] = dict(self.mask_paths)
        volume_index['volumes'] = sorted(self.image_paths.keys())
        if not volume_index['volumes'] and self.strict:
            raise ValueError(f"No Volume-*.png files found in {self.image_dir}")
        for vid in volume_index['volumes']:
            image_ids = {self.extract_slice_id(p.name) for p in self.image_paths[vid]}
            mask_ids = {self.extract_slice_id(p.name) for p in self.mask_paths.get(vid, [])}
            if image_ids != mask_ids:
                missing = sorted(image_ids - mask_ids)[:10]
                extra = sorted(mask_ids - image_ids)[:10]
                raise ValueError(
                    f"Image/mask slice mismatch for volume {vid}: "
                    f"missing masks={missing}, extra masks={extra}"
                )
        return volume_index


class VolumeWiseSplitter:
    def __init__(self, split_ratios=(0.8, 0.1, 0.1)):
        self.split_ratios = split_ratios
        self.train_ratio = split_ratios[0]
        self.val_ratio = split_ratios[0] + split_ratios[1]

    def split(self, volume_ids: List[int]) -> Dict[str, List[int]]:
        sorted_ids = sorted(volume_ids)
        n = len(sorted_ids)
        train_end = int(n * self.train_ratio)
        val_end = int(n * self.val_ratio)
        return {
            'train': sorted_ids[:train_end],
            'val': sorted_ids[train_end:val_end],
            'test': sorted_ids[val_end:],
        }

    def load_splits(self, splits_dir: Path) -> Dict[str, List[int]]:
        splits = {}
        for split_name in ['train', 'val', 'test']:
            filepath = splits_dir / f"{split_name}_volumes.txt"
            if not filepath.exists():
                splits[split_name] = []
                continue
            with open(filepath, 'r') as f:
                content = f.read().strip()
            if ',' in content:
                splits[split_name] = [int(x.strip()) for x in content.split(',') if x.strip()]
            else:
                splits[split_name] = [int(line.strip()) for line in content.split('\n') if line.strip()]
        return splits

    def save_splits(self, splits: Dict[str, List[int]], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, vol_ids in splits.items():
            filepath = output_dir / f"{split_name}_volumes.txt"
            with open(filepath, 'w') as f:
                for vid in sorted(vol_ids):
                    f.write(f"{vid}\n")


@DATASETS.register("lits")
class LiverTumor2DDataset(Dataset):
    def __init__(self, volume_ids: List[int], volume_index: Dict, transform=None):
        self.volume_ids = volume_ids
        self.volume_index = volume_index
        self.transform = transform
        self.samples = []
        for vid in volume_ids:
            vid_int = int(vid) if not isinstance(vid, int) else vid
            if vid_int in volume_index.get('image_paths', {}):
                n_slices = len(volume_index['image_paths'][vid_int])
                for sid in range(n_slices):
                    self.samples.append((vid_int, sid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        vid, sid = self.samples[idx]
        img_path = self.volume_index['image_paths'][vid][sid]
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0

        if vid in self.volume_index.get('mask_paths', {}):
            mask_sid = min(sid, len(self.volume_index['mask_paths'][vid]) - 1)
            mask_path = self.volume_index['mask_paths'][vid][mask_sid]
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask, dtype=np.float32)
            mask = (mask > 0.5).astype(np.float32)
        else:
            mask = np.zeros_like(img)

        if self.transform:
            img, mask = self.transform(img, mask)
        elif mask.shape != img.shape:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((img.shape[1], img.shape[0]), Image.Resampling.NEAREST)
            mask = (np.asarray(mask_img) > 127).astype(np.float32)

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'volume_id': vid,
            'slice_id': sid,
        }


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
    tumor_sampler_weight: Optional[float] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = LiverTumor2DDataset(train_vids, volume_index, transform=transform_train)
    val_ds = LiverTumor2DDataset(val_vids, volume_index, transform=transform_val)
    test_ds = LiverTumor2DDataset(test_vids, volume_index, transform=transform_val)

    sampler = None
    if tumor_sampler_weight is not None and len(train_ds) > 0:
        weights = []
        for vid, sid in train_ds.samples:
            mask_path = volume_index['mask_paths'][vid][sid]
            with Image.open(mask_path) as mask_img:
                has_tumor = mask_img.convert('L').getbbox() is not None
            weights.append(float(tumor_sampler_weight if has_tumor else 1.0))
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True, generator=generator
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(sampler is None and len(train_ds) > 0),
        sampler=sampler,
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
    return train_loader, val_loader, test_loader

from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.framework.core.config import build_experiment_config
from src.framework.core.factory import build_loss, build_model
from src.framework.data.lits_dataset import (
    DataPathManager, VolumeWiseSplitter, create_2d_dataloaders,
)
from src.framework.data.manifest_dataset import VerifiedManifestDataset, create_manifest_dataloaders
from src.framework.data.transforms import (
    AugmentedPreprocessingTransform, PreprocessingTransform,
)


DEFAULT_IMAGES = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset")
DEFAULT_MASKS = Path(r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks")


def configure_dataset_environment(images_dir: Optional[str] = None,
                                  masks_dir: Optional[str] = None) -> Tuple[str, str]:
    images = images_dir or os.getenv("MEDSEGX_LITS_IMAGES_DIR")
    masks = masks_dir or os.getenv("MEDSEGX_LITS_MASKS_DIR")
    if not images and DEFAULT_IMAGES.is_dir():
        images = str(DEFAULT_IMAGES)
    if not masks and DEFAULT_MASKS.is_dir():
        masks = str(DEFAULT_MASKS)
    if not images or not masks:
        raise RuntimeError(
            "Set MEDSEGX_LITS_IMAGES_DIR and MEDSEGX_LITS_MASKS_DIR or pass dataset paths"
        )
    os.environ["MEDSEGX_LITS_IMAGES_DIR"] = images
    os.environ["MEDSEGX_LITS_MASKS_DIR"] = masks
    return images, masks


def load_experiment(path: str, images_dir: Optional[str] = None,
                    masks_dir: Optional[str] = None) -> Dict:
    configure_dataset_environment(images_dir, masks_dir)
    return build_experiment_config(path)


def build_experiment_model(cfg: Dict, pretrained: Optional[bool] = None):
    model_cfg = dict(cfg["model"])
    if pretrained is not None:
        model_cfg["pretrained"] = pretrained
    return build_model(model_cfg)


def build_experiment_loss(cfg: Dict):
    loss_cfg = dict(cfg["loss"])
    name = loss_cfg["name"]
    # Inherited YAML mappings are deep-merged, so a loss replacement may retain
    # parameters from its parent. Filter those deterministically by constructor.
    from src.framework.core.registry import LOSSES
    constructor = LOSSES.get(name)
    accepted = set(inspect.signature(constructor.__init__).parameters) - {"self"}
    filtered = {key: value for key, value in loss_cfg.items()
                if key == "name" or key in accepted}
    return build_loss(filtered)


def _manifest_volume_index(manifest_path) -> Dict:
    import csv
    volumes = set()
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                volumes.add(int(row["volume_id"]))
            except (KeyError, ValueError, TypeError):
                continue
    return {"volumes": sorted(volumes)}


def build_experiment_loaders(cfg: Dict, include_sampler: bool = True,
                             limit_volumes: Optional[int] = None,
                             limit_slices: Optional[int] = None):
    dataset_cfg = cfg["dataset"]
    manifest_path = dataset_cfg.get("slice_manifest")
    if manifest_path:
        return _build_manifest_loaders(
            cfg, include_sampler=include_sampler, limit_slices=limit_slices,
        )
    manager = DataPathManager(
        dataset_cfg["images_dir"], dataset_cfg["masks_dir"], strict=True
    )
    index = manager.build_index()
    split_dir = Path(cfg.get("data", {}).get("split_dir", "data/splits"))
    splits = VolumeWiseSplitter().load_splits(split_dir)
    if limit_volumes:
        splits = {key: values[:limit_volumes] for key, values in splits.items()}
    preprocessing = cfg["preprocessing"]
    augmentation = cfg.get("augmentation", {})
    target_size = tuple(preprocessing.get("target_size", [256, 256]))
    hu_low, hu_high = preprocessing.get("hu_window", [-100, 400])
    apply_hu = preprocessing.get("apply_hu_window", False)
    train_transform = AugmentedPreprocessingTransform(
        target_size=target_size, hu_low=hu_low, hu_high=hu_high,
        clahe=None, flip_prob=augmentation.get("horizontal_flip_prob", 0.5),
        shift_range=augmentation.get("intensity_shift_range", 0.1),
        apply_hu_window=apply_hu,
    )
    eval_transform = PreprocessingTransform(
        target_size=target_size, hu_low=hu_low, hu_high=hu_high,
        clahe=None, apply_hu_window=apply_hu,
    )
    training = cfg["training"]
    loaders = create_2d_dataloaders(
        index, splits["train"], splits["val"], splits["test"],
        batch_size=training.get("batch_size", 4),
        num_workers=training.get("num_workers", 0),
        pin_memory=training.get("pin_memory", True),
        transform_train=train_transform, transform_val=eval_transform,
        tumor_sampler_weight=(training.get("tumor_sampler_weight") if include_sampler else None),
        seed=cfg["experiment"].get("seed", 42),
    )
    return loaders, index, splits, split_dir


def _build_manifest_loaders(cfg: Dict, include_sampler: bool = True,
                            limit_slices: Optional[int] = None):
    dataset_cfg = cfg["dataset"]
    root_dir = dataset_cfg["root_dir"]
    manifest_path = dataset_cfg["slice_manifest"]
    preprocessing = cfg["preprocessing"]
    augmentation = cfg.get("augmentation", {})
    target_size = tuple(preprocessing.get("target_size", [256, 256]))
    hu_low, hu_high = preprocessing.get("hu_window", [-100, 400])
    apply_hu = preprocessing.get("apply_hu_window", False)
    train_transform = AugmentedPreprocessingTransform(
        target_size=target_size, hu_low=hu_low, hu_high=hu_high,
        clahe=None, flip_prob=augmentation.get("horizontal_flip_prob", 0.5),
        shift_range=augmentation.get("intensity_shift_range", 0.1),
        apply_hu_window=apply_hu,
    )
    eval_transform = PreprocessingTransform(
        target_size=target_size, hu_low=hu_low, hu_high=hu_high,
        clahe=None, apply_hu_window=apply_hu,
    )
    training = cfg["training"]
    train_loader, val_loader, _ = create_manifest_dataloaders(
        manifest_path, root_dir=root_dir,
        batch_size=training.get("batch_size", 4),
        num_workers=training.get("num_workers", 0),
        pin_memory=training.get("pin_memory", True),
        transform_train=train_transform, transform_val=eval_transform,
        tumor_sampler_weight=(
            training.get("tumor_sampler_weight") if include_sampler else None
        ),
        seed=cfg["experiment"].get("seed", 42),
        limit_slices=limit_slices,
    )
    split_dir = Path(dataset_cfg.get("split_dir", "data/splits"))
    splits = VolumeWiseSplitter().load_splits(split_dir)
    index = _manifest_volume_index(manifest_path)
    return (train_loader, val_loader, None), index, splits, split_dir

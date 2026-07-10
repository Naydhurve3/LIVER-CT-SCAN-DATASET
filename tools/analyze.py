from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.framework.evaluation.checkpoints import split_hashes, write_json
from src.framework.experiment import configure_dataset_environment
from src.framework.data.lits_dataset import DataPathManager, VolumeWiseSplitter


def parse_args():
    parser = argparse.ArgumentParser(description="Audit LiTS PNG paths and volume splits")
    parser.add_argument("--images-dir")
    parser.add_argument("--masks-dir")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--output", default="experiments/research_validation/data_audit.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images, masks = configure_dataset_environment(args.images_dir, args.masks_dir)
    index = DataPathManager(images, masks, strict=True).build_index()
    splits = VolumeWiseSplitter().load_splits(Path(args.split_dir))
    sets = {name: set(values) for name, values in splits.items()}
    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    image_count = sum(len(paths) for paths in index["image_paths"].values())
    mask_count = sum(len(paths) for paths in index["mask_paths"].values())
    result = {
        "volumes": len(index["volumes"]), "images": image_count, "masks": mask_count,
        "matched_slice_count": image_count == mask_count,
        "split_volume_counts": {name: len(values) for name, values in splits.items()},
        "split_overlap": overlap, "no_split_overlap": not any(overlap.values()),
        "split_hashes": split_hashes(args.split_dir),
        "acceptance": image_count == 58638 and mask_count == 58638 and
                      len(index["volumes"]) == 131 and not any(overlap.values()),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

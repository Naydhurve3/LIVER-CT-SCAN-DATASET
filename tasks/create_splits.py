"""Create stratified train/val/test splits from the volume index."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import DatasetConfig
from src.data_loader import DataPathManager, VolumeWiseSplitter


def run(output_dir=None, val_ratio=0.1, test_ratio=0.1, seed=42):
    output = Path(output_dir or DatasetConfig.SPLITS_DIR)
    output.mkdir(parents=True, exist_ok=True)

    path_manager = DataPathManager()
    volume_index = path_manager.build_index()

    splitter = VolumeWiseSplitter(split_ratios=(1 - val_ratio - test_ratio, val_ratio, test_ratio))
    splits = splitter.split(volume_index['volumes'])
    splitter.save_splits(splits, output)

    for name, vids in splits.items():
        n_slices = sum(len(volume_index['image_paths'].get(v, [])) for v in vids)
        print(f"  {name}: {len(vids)} volumes, ~{n_slices} slices")

    print(f"Splits saved to {output}")
    return splits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create stratified data splits")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(output_dir=args.output, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

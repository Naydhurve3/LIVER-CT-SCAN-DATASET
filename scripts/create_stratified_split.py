"""S4: Stratified split per slice preserving tumor distribution.
Creates slice-level train/val/test splits with equal tumor-positive ratios.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils import setup_logging, logger
from src.data_loader import DataPathManager


def create_split(test_size=0.15, val_size=0.15, random_state=42):
    mgr = DataPathManager()
    index = mgr.build_index()
    slices = []
    for vid in index['volumes']:
        mask_paths = index['mask_paths'].get(vid, [])
        img_paths = index['image_paths'].get(vid, [])
        for mp, ip in zip(mask_paths, img_paths):
            m = np.array(Image.open(mp).convert('L'), dtype=np.float32)
            has_tumor = int((m > 0.5).sum() > 0)
            slices.append({
                "volume_id": vid, "image_path": str(ip),
                "mask_path": str(mp), "has_tumor": has_tumor,
            })

    labels = np.array([s["has_tumor"] for s in slices])
    n_tumor = labels.sum()
    n_total = len(labels)
    logger.info(f"Total slices: {n_total}, tumor-positive: {n_tumor} ({n_tumor/n_total*100:.1f}%)")

    # Stratified split: first separate test, then split remaining into train/val
    idx = np.arange(n_total)
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=random_state)
    relabel = labels[train_val_idx]
    val_frac = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac, stratify=relabel, random_state=random_state)

    splits = {
        "train": [slices[i] for i in sorted(train_idx)],
        "val": [slices[i] for i in sorted(val_idx)],
        "test": [slices[i] for i in sorted(test_idx)],
    }

    out_dir = Path(__file__).resolve().parent.parent / "data" / "splits_stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, s in splits.items():
        paths = [{"volume": x["volume_id"], "image": x["image_path"], "mask": x["mask_path"]}
                 for x in s]
        n_t = sum(x["has_tumor"] for x in s)
        summary[name] = {
            "slices": len(s), "tumor_slices": int(n_t),
            "tumor_pct": round(n_t / len(s) * 100, 1),
        }
        (out_dir / f"{name}.json").write_text(json.dumps(paths, indent=2))
        (out_dir / f"{name}.txt").write_text("\n".join(x["image_path"] for x in s))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    logger.info(f"Stratified split saved to {out_dir}")
    return splits, summary


if __name__ == "__main__":
    setup_logging()
    create_split()

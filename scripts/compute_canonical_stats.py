"""S2: Compute full-population canonical statistics on all 58,638 slices.
Uses numpy vectorized operations for performance (~2 min on default 256x256 images).
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm
from src.utils import setup_logging, logger
from src.data_loader import DataPathManager


def compute():
    mgr = DataPathManager()
    index = mgr.build_index()
    all_imgs = [(vid, p) for vid in index['volumes'] for p in index['image_paths'].get(vid, [])]
    all_masks = [(vid, p) for vid in index['volumes'] for p in index['mask_paths'].get(vid, [])]
    assert len(all_imgs) == len(all_masks) == 58638

    # Streaming accumulators (numpy vectorized, no per-pixel loops)
    sum_px = 0.0
    sum_sq_px = 0.0
    n_px = 0
    pixel_min = float('inf')
    pixel_max = float('-inf')

    tumor_px_total = 0
    n_slices_total = len(all_imgs)
    n_tumor_slices = 0

    per_vol = {str(vid): {"slices": 0, "slices_with_tumor": 0, "total_tumor_pixels": 0}
               for vid in index['volumes']}

    # Per-image percentile accumulators (average, avoids OOM on concat)
    q1_vals, med_vals, q3_vals = [], [], []

    for (vid, img_path), (_, mask_path) in tqdm(zip(all_imgs, all_masks), total=n_slices_total, desc="Canonical stats"):
        try:
            img = np.array(Image.open(img_path), dtype=np.float32).ravel()
            mask = np.array(Image.open(mask_path), dtype=np.uint8).ravel()
        except Exception as e:
            logger.warning(f"Skipping {img_path}: {e}")
            continue

        # Intensity stats: vectorized
        sum_px += img.sum()
        sum_sq_px += (img ** 2).sum()
        n_px += img.size
        pmax = img.max()
        pmin = img.min()
        if pmax > pixel_max: pixel_max = float(pmax)
        if pmin < pixel_min: pixel_min = float(pmin)

        # Tumor stats
        tp = int((mask > 0.5).sum())
        tumor_px_total += tp
        vid_s = str(vid)
        per_vol[vid_s]["slices"] += 1
        per_vol[vid_s]["total_tumor_pixels"] += tp
        if tp > 0:
            n_tumor_slices += 1
            per_vol[vid_s]["slices_with_tumor"] += 1

        # Per-image percentiles (averaged across all slices)
        q1_vals.append(float(np.percentile(img, 25)))
        med_vals.append(float(np.median(img)))
        q3_vals.append(float(np.percentile(img, 75)))

    # Final stats
    mean = sum_px / n_px
    std = np.sqrt(sum_sq_px / n_px - mean ** 2)
    for v in per_vol.values():
        v["tumor_slice_pct"] = round(v["slices_with_tumor"] / max(v["slices"], 1) * 100, 1)

    # Average per-image percentiles
    q1 = float(np.mean(q1_vals)) if q1_vals else 0.0
    med = float(np.mean(med_vals)) if med_vals else 0.0
    q3 = float(np.mean(q3_vals)) if q3_vals else 0.0

    bg_px = n_px - tumor_px_total
    stats = {
        "phase": "Phase 4 - Canonical Statistics (Full Population)",
        "date": "2026-07-08",
        "image_size": [256, 256],
        "dataset_summary": {"total_volumes": len(index['volumes']), "total_slices": n_slices_total},
        "intensity_statistics": {
            "mean": round(float(mean), 6), "std": round(float(std), 6),
            "min": pixel_min, "q1": q1, "median": med, "q3": q3, "max": pixel_max,
            "samples": int(n_px),
        },
        "tumor_statistics": {
            "total_slices": n_slices_total,
            "slices_with_tumor": n_tumor_slices,
            "slices_without_tumor": n_slices_total - n_tumor_slices,
            "tumor_slice_pct": round(n_tumor_slices / n_slices_total * 100, 2),
            "total_tumor_pixels": tumor_px_total,
            "total_background_pixels": bg_px,
            "tumor_pixel_pct": round(tumor_px_total / n_px * 100, 4),
            "background_pct": round(bg_px / n_px * 100, 4),
            "imbalance_ratio": round(bg_px / max(tumor_px_total, 1), 2),
            "per_volume": per_vol,
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "data" / "metadata" / "statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"Canonical stats saved to {out_path}")

    print(json.dumps(stats["intensity_statistics"], indent=2))
    ts = stats["tumor_statistics"]
    print(f"\nTumor: {ts['slices_with_tumor']}/{ts['total_slices']} slices ({ts['tumor_slice_pct']}%)")
    print(f"Pixels: {ts['tumor_pixel_pct']}% tumor, imbalance {ts['imbalance_ratio']}:1")
    return stats


if __name__ == "__main__":
    setup_logging()
    compute()

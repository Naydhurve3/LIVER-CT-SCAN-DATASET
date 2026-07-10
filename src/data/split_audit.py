"""Split composition audit — zero-tumor / low-burden volume analysis."""
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def load_volume_manifest(
    images_dir: str,
    masks_dir: str,
    splits_dir: str,
) -> pd.DataFrame:
    """One row per volume: volume_id, split, n_slices,
    tumor_positive_slices, tumor_pixel_count, tumor_burden_pct."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    splits_dir = Path(splits_dir)

    # Load split assignments
    split_map = {}
    for split_name in ["train", "val", "test"]:
        fp = splits_dir / f"{split_name}_volumes.txt"
        if not fp.exists():
            continue
        for line in fp.read_text().strip().splitlines():
            line = line.strip()
            if line:
                split_map[int(line)] = split_name

    # Group masks by volume
    vol_masks: Dict[int, List[Path]] = defaultdict(list)
    for mp in sorted(masks_dir.glob("mask-*.png")):
        match = __import__("re").match(r"mask-(\d+)-\d+\.png", mp.name)
        if match:
            vol_masks[int(match.group(1))].append(mp)

    if not vol_masks:
        vol_masks = defaultdict(list)
        for mp in sorted(masks_dir.glob("*.png")):
            match = __import__("re").match(r"mask-(\d+)-\d+\.png", mp.name)
            if match:
                vol_masks[int(match.group(1))].append(mp)

    rows = []
    for vid in sorted(vol_masks):
        n_slices = len(vol_masks[vid])
        tumor_slices = 0
        tumor_px = 0
        total_px = 0
        for mp in vol_masks[vid]:
            m = np.array(Image.open(mp))
            if m.ndim == 3:
                m = m[:, :, 0]
            pos = (m > 0).sum()
            if pos > 0:
                tumor_slices += 1
            tumor_px += int(pos)
            # LiTS PNG images are 512x512 while masks are 256x256 RGB.
            # Define burden in native mask coordinates to avoid a 4x error.
            total_px += m.size
        burden_pct = 100.0 * tumor_px / max(total_px, 1)
        rows.append(
            {
                "volume_id": vid,
                "split": split_map.get(vid, "unknown"),
                "n_slices": n_slices,
                "tumor_positive_slices": tumor_slices,
                "tumor_pixel_count": tumor_px,
                "tumor_burden_pct": round(burden_pct, 4),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("volume_id").reset_index(drop=True)
    return df


def flag_zero_tumor_volumes(
    df: pd.DataFrame, burden_low_threshold: float = 0.05
) -> pd.DataFrame:
    """Adds is_zero_tumor and is_low_burden columns."""
    df = df.copy()
    df["is_zero_tumor"] = df["tumor_positive_slices"] == 0
    df["is_low_burden"] = (df["tumor_burden_pct"] < burden_low_threshold) & (
        df["tumor_positive_slices"] > 0
    )
    return df


def split_composition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per split: n_volumes, n_zero_tumor, pct_zero_tumor,
    median_tumor_burden, mean_tumor_burden."""
    rows = []
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        if len(subset) == 0:
            continue
        n_zero = subset["is_zero_tumor"].sum()
        n_low = subset["is_low_burden"].sum()
        rows.append(
            {
                "split": split_name,
                "n_volumes": len(subset),
                "n_zero_tumor": int(n_zero),
                "pct_zero_tumor": round(100.0 * n_zero / len(subset), 1),
                "n_low_burden": int(n_low),
                "pct_low_burden": round(100.0 * n_low / len(subset), 1),
                "median_tumor_burden_pct": round(
                    subset["tumor_burden_pct"].median(), 4
                ),
                "mean_tumor_burden_pct": round(
                    subset["tumor_burden_pct"].mean(), 4
                ),
                "max_tumor_burden_pct": round(
                    subset["tumor_burden_pct"].max(), 4
                ),
            }
        )
    return pd.DataFrame(rows)


def save_audit_outputs(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str = "experiments/sprint1",
) -> None:
    """Save volume-level and per-split summary CSVs.

    Creates:
      output_dir/split_audit_volumes.csv  — one row per volume (131 rows)
      output_dir/split_audit_summary.csv  — one row per split (3 rows)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vol_path = out / "split_audit_volumes.csv"
    df.to_csv(vol_path, index=False)
    print(f"  [saved] {vol_path} ({len(df)} rows)")

    sum_path = out / "split_audit_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"  [saved] {sum_path} ({len(summary_df)} rows)")


def plot_tumor_burden_by_split(df: pd.DataFrame) -> None:
    """Boxplot/strip plot of tumor_burden_pct grouped by split, log-scale y-axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = df[df["split"].isin(["train", "val", "test"])].copy()
    plot_df["split"] = pd.Categorical(
        plot_df["split"], categories=["train", "val", "test"], ordered=True
    )
    # Add jittered strip plot
    for i, split_name in enumerate(["train", "val", "test"]):
        subset = plot_df[plot_df["split"] == split_name]
        y = subset["tumor_burden_pct"].values + 1e-6
        x = np.full_like(y, i) + np.random.RandomState(42).uniform(
            -0.15, 0.15, size=len(y)
        )
        ax.scatter(x, y, alpha=0.6, s=30, edgecolors="k", linewidth=0.5, zorder=3)
    bp = ax.boxplot(
        [plot_df[plot_df["split"] == s]["tumor_burden_pct"].values + 1e-6 for s in ["train", "val", "test"]],
        positions=[0, 1, 2],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    ax.set_yscale("symlog", linthresh=0.001)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Train", "Val", "Test"])
    ax.set_ylabel("Tumor Burden (% of volume pixels)")
    ax.set_title("Tumor Burden Distribution by Split")
    ax.axhline(y=0.05, color="gray", ls="--", alpha=0.6, label="Low-burden threshold (0.05%)")
    ax.legend()
    plt.tight_layout()
    plt.show()

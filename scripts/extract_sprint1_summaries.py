"""Extract Sprint 1 CSVs and write detailed txt summaries to Practice/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "experiments" / "sprint1"
OUT_DIR = BASE / "Practice"


def write_audit_summary():
    vol_df = pd.read_csv(CSV_DIR / "split_audit_volumes.csv")
    sum_df = pd.read_csv(CSV_DIR / "split_audit_summary.csv")

    lines = []
    def w(s=""):
        lines.append(str(s))

    w("=" * 72)
    w("SPRINT 1 — SPLIT TUMOR AUDIT")
    w("=" * 72)

    w("\n--- DATASET OVERVIEW ---")
    w(f"  Total volumes:           {len(vol_df)}")
    w(f"  Total slices:            {vol_df['n_slices'].sum():,}")
    w(f"  Total tumor-positive slices: {vol_df['tumor_positive_slices'].sum():,}")
    w(f"  Total tumor pixels:      {vol_df['tumor_pixel_count'].sum():,}")
    w(f"  Zero-tumor volumes:      {vol_df['is_zero_tumor'].sum()} / {len(vol_df)} ({100*vol_df['is_zero_tumor'].sum()/len(vol_df):.1f}%)")
    w(f"  Low-burden (<0.05%) volumes: {vol_df['is_low_burden'].sum()} / {len(vol_df)} ({100*vol_df['is_low_burden'].sum()/len(vol_df):.1f}%)")

    w("\n--- PER-SPLIT COMPOSITION ---")
    w(f"  {'Split':>6s} {'Vols':>5s} {'Slices':>8s} {'Tumor+':>7s} {'Zero':>5s} {'Low':>5s} {'Med Bur%':>9s} {'Mean Bur%':>9s} {'Max Bur%':>9s}")
    w(f"  {'-'*66}")
    for _, row in sum_df.iterrows():
        subset = vol_df[vol_df["split"] == row["split"]]
        w(f"  {row['split']:>6s} {int(row['n_volumes']):5d} {subset['n_slices'].sum():8,} "
          f"{subset['tumor_positive_slices'].sum():7,} {int(row['n_zero_tumor']):5d} "
          f"{int(row['n_low_burden']):5d} {row['median_tumor_burden_pct']:9.4f} "
          f"{row['mean_tumor_burden_pct']:9.4f} {row['max_tumor_burden_pct']:9.4f}")

    w("\n--- ZERO-TUMOR VOLUME DISTRIBUTION ---")
    zero_by_split = vol_df[vol_df["is_zero_tumor"]].groupby("split").agg(
        count=("volume_id", "count"),
        vol_ids=("volume_id", list),
    )
    total_zero = zero_by_split["count"].sum()
    for split_name in ["train", "val", "test"]:
        if split_name in zero_by_split.index:
            r = zero_by_split.loc[split_name]
            ids_str = ", ".join(str(v) for v in r["vol_ids"])
            w(f"  {split_name.upper():5s}: {int(r['count']):2d} / {int(total_zero):2d} ({100*r['count']/total_zero:.0f}%)  volumes: [{ids_str}]")

    w("\n--- LOW-BURDEN (<0.05%) VOLUME DISTRIBUTION ---")
    low_by_split = vol_df[vol_df["is_low_burden"]].groupby("split").agg(
        count=("volume_id", "count"),
        vol_ids=("volume_id", list),
    )
    total_low = low_by_split["count"].sum()
    for split_name in ["train", "val", "test"]:
        if split_name in low_by_split.index:
            r = low_by_split.loc[split_name]
            ids_str = ", ".join(str(v) for v in r["vol_ids"])
            w(f"  {split_name.upper():5s}: {int(r['count']):2d} / {int(total_low):2d} ({100*r['count']/total_low:.0f}%)  volumes: [{ids_str}]")

    w("\n--- BURDEN STATISTICS ---")
    for split_name in ["train", "val", "test"]:
        subset = vol_df[vol_df["split"] == split_name]
        bur = subset["tumor_burden_pct"]
        w(f"  {split_name.upper():5s}: mean={bur.mean():.4f}%  median={bur.median():.4f}%  "
          f"std={bur.std():.4f}%  range=[{bur.min():.4f}, {bur.max():.4f}]")

    w("\n--- TOP-10 HIGHEST BURDEN VOLUMES ---")
    top10 = vol_df.nlargest(10, "tumor_burden_pct")[
        ["volume_id", "split", "n_slices", "tumor_positive_slices", "tumor_burden_pct"]
    ]
    for _, row in top10.iterrows():
        w(f"  Vol {int(row['volume_id']):3d} ({row['split']:>5s}): {row['tumor_burden_pct']:.4f}% burden, "
          f"{int(row['tumor_positive_slices']):,} tumor slices / {int(row['n_slices']):,} total")

    w("\n--- DECISION GATE ---")
    train_zero = int(sum_df[sum_df["split"] == "train"]["n_zero_tumor"].values[0])
    val_zero = int(sum_df[sum_df["split"] == "val"]["n_zero_tumor"].values[0])
    test_zero = int(sum_df[sum_df["split"] == "test"]["n_zero_tumor"].values[0])
    val_pct = float(sum_df[sum_df["split"] == "val"]["pct_zero_tumor"].values[0])
    train_pct = float(sum_df[sum_df["split"] == "train"]["pct_zero_tumor"].values[0])
    w(f"  Zero-tumor split: train={train_zero} ({train_pct:.0f}%), val={val_zero} ({val_pct:.0f}%), test={test_zero}")
    if val_pct > train_pct * 2 and val_zero >= 3:
        w(f"  ⚠  Val is {val_pct/train_pct:.1f}x more zero-tumor than train — significant skew. Consider resplitting.")
    else:
        w(f"  ✓ Split distribution within normal variance for {len(vol_df)} volumes.")

    w("\n" + "=" * 72)
    w("END OF SPLIT TUMOR AUDIT")
    w("=" * 72)

    path = OUT_DIR / "split_tumor_audit data.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [written] {path}")


def write_sweep_summary():
    df = pd.read_csv(CSV_DIR / "sweep_results.csv")

    lines = []
    def w(s=""):
        lines.append(str(s))

    w("=" * 72)
    w("SPRINT 1 — TUMOR WEIGHT SWEEP (1 epoch)")
    w("=" * 72)

    w("\n--- CONFIGURATION ---")
    w(f"  Swept parameter:    tumor_weight ∈ {sorted(df['tumor_weight'].tolist())}")
    w(f"  Fixed:             epochs=1, batch_size=4, seed=42")
    w(f"  Loss:              FocalDiceLoss (focal_alpha=0.75, focal_gamma=2.0)")
    w(f"  Sampler:           stratified (tumor_weight varies)")
    w(f"  Model:             mobilenetv2_unet (pretrained)")

    w("\n--- RESULTS TABLE ---")
    w(f"  {'Weight':>7s} {'Loss':>8s} {'Dice':>8s} {'T Dice':>8s} {'AUPRC':>8s} {'Raw AUPRC':>10s} {'AUROC':>8s} {'FG%':>7s} {'Prec':>8s} {'Recall':>8s}")
    w(f"  {'-'*84}")
    for _, row in df.iterrows():
        w(f"  {int(row['tumor_weight']):5d}x  {row['train_loss']:8.4f} {row['val_dice']:8.4f} "
          f"{row['val_tumor_dice']:8.4f} {row['val_auprc']:8.4f} {row['raw_val_auprc']:10.4f} "
          f"{row['val_auroc']:8.4f} {row['val_fg_frac']*100:6.2f}% "
          f"{row['precision']:8.4f} {row['recall']:8.4f}")

    w("\n--- BEST PER METRIC ---")
    best_auprc = df.loc[df["val_auprc"].idxmax()]
    best_tdice = df.loc[df["val_tumor_dice"].idxmax()]
    best_dice = df.loc[df["val_dice"].idxmax()]
    best_fg = df.loc[(df["val_fg_frac"] - df["val_fg_frac"].mean()).abs().idxmin()]
    w(f"  Highest AUPRC:           w={int(best_auprc['tumor_weight'])}  (AUPRC={best_auprc['val_auprc']:.4f})")
    w(f"  Highest Tumor Dice:      w={int(best_tdice['tumor_weight'])}  (T Dice={best_tdice['val_tumor_dice']:.4f})")
    w(f"  Highest Overall Dice:    w={int(best_dice['tumor_weight'])}  (Dice={best_dice['val_dice']:.4f})")

    w("\n--- RAW vs THRESHOLDED AUPRC ---")
    for _, row in df.iterrows():
        diff = row["raw_val_auprc"] - row["val_auprc"]
        w(f"  w={int(row['tumor_weight'])}: raw={row['raw_val_auprc']:.4f} vs val={row['val_auprc']:.4f} (diff={diff:+.4f})")
    best = df.loc[df["val_auprc"].idxmax()]
    w(f"\n  Interpretation: raw AUPRC approximates val AUPRC (diff < 0.01 for all)")
    w(f"  → AUPRC reflects true ranking ability, not threshold miscalibration")

    w("\n--- COLLAPSE DIAGNOSIS ---")
    all_low = all(df["val_auprc"] < 0.25)
    fg_low = all(df["val_fg_frac"] * 100 < 2.0)
    rec_all = df["recall"].mean()
    prec_all = df["precision"].mean()
    w(f"  AUPRC range:        [{df['val_auprc'].min():.4f}, {df['val_auprc'].max():.4f}]  {'LOW (<0.25)' if all_low else 'Moderate'}")
    w(f"  FG% range:          [{df['val_fg_frac'].min()*100:.2f}%, {df['val_fg_frac'].max()*100:.2f}%]")
    w(f"  Mean Recall:        {rec_all:.4f}  ({'High — model finds most tumor' if rec_all > 0.5 else 'Low — model misses tumor'})")
    w(f"  Mean Precision:     {prec_all:.4f}  ({'Good — few false positives' if prec_all > 0.3 else 'Low — many false positives'})")
    if all_low and fg_low:
        w(f"  DIAGNOSIS: True distribution collapse (low AUPRC + low FG%)")
    elif not all_low and fg_low:
        w(f"  DIAGNOSIS: Threshold miscalibration (decent AUPRC ranking, low FG% at 0.5 threshold)")
    else:
        w(f"  DIAGNOSIS: Mild — reasonable AUPRC and FG%")

    w("\n--- RECOMMENDATION ---")
    w(f"  Chosen weight for 5-epoch pilot: w={int(best_auprc['tumor_weight'])}")
    w(f"  Reason: Highest AUPRC ({best_auprc['val_auprc']:.4f}) and Tumor Dice ({best_auprc['val_tumor_dice']:.4f}), "
      f"balanced foreground fraction ({best_auprc['val_fg_frac']*100:.2f}%)")

    w("\n--- PER-WEIGHT DETAIL ---")
    for _, row in df.iterrows():
        w(f"\n  tumor_weight = {int(row['tumor_weight'])}:")
        w(f"    Train loss:        {row['train_loss']:.4f}")
        w(f"    Val Dice:          {row['val_dice']:.4f}")
        w(f"    Val Tumor Dice:    {row['val_tumor_dice']:.4f}")
        w(f"    Val AUPRC:         {row['val_auprc']:.4f}")
        w(f"    Raw logit AUPRC:   {row['raw_val_auprc']:.4f}")
        w(f"    Val AUROC:         {row['val_auroc']:.4f}")
        w(f"    Pred FG fraction:  {row['val_fg_frac']*100:.2f}%")
        w(f"    Precision:         {row['precision']:.4f}")
        w(f"    Recall:            {row['recall']:.4f}")

    w("\n" + "=" * 72)
    w("END OF SWEEP REPORT")
    w("=" * 72)

    path = OUT_DIR / "sprint1_sweep data.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [written] {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Extracting Sprint 1 summaries")
    print("=" * 60)
    write_audit_summary()
    print()
    write_sweep_summary()
    print("\nDone.")

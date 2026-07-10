"""Thin wrapper to run short 1-epoch config sweeps."""
import os, json, copy
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
import pandas as pd

from src.config import DEVICE
from src.models import create_model, count_params
from src.data_loader import (
    DatasetConfig,
    DataPathManager,
    VolumeWiseSplitter,
    create_2d_dataloaders,
)
from src.preprocessing import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.trainer import Trainer
from src.metrics import slice_level_auprc


def compute_val_auprc(model, val_loader) -> float:
    """Precision-recall AUC on raw validation logits, NOT thresholded predictions."""
    model.eval()
    device = next(model.parameters()).device
    all_probs, all_masks = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = torch.sigmoid(model(images))
            all_probs.append(preds.cpu())
            all_masks.append(masks.cpu())
    all_probs_t = torch.cat(all_probs, dim=0)
    all_masks_t = torch.cat(all_masks, dim=0)
    return slice_level_auprc(all_probs_t, all_masks_t)


def run_single_config(
    tumor_weight: float,
    epochs: int = 1,
    output_dir: str = "experiments/sprint1",
    seed: int = 42,
    batch_size: int = 4,
) -> dict:
    """Run training for `epochs`, return dict of final-epoch metrics."""
    import warnings
    warnings.filterwarnings("ignore")
    from src.utils import set_seed
    set_seed(seed)

    output_dir = Path(output_dir)
    run_dir = output_dir / f"sweep_w{int(tumor_weight)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    path_manager = DataPathManager()
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(DatasetConfig.SPLITS_DIR)

    clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
    transform_train = AugmentedPreprocessingTransform((256, 256), -100, 400, clahe)
    transform_val = PreprocessingTransform((256, 256), -100, 400)

    train_loader, val_loader, _ = create_2d_dataloaders(
        volume_index,
        splits["train"],
        splits["val"],
        splits["test"],
        batch_size=batch_size,
        num_workers=0,
        transform_train=transform_train,
        transform_val=transform_val,
        use_tumor_sampler=True,
        tumor_sampler_weight=tumor_weight,
    )

    model = create_model(
        "mobilenetv2_unet", in_channels=1, out_channels=1, pretrained=True
    ).to(DEVICE)

    train_config = {
        "input_size": (256, 256),
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": False,
        "lr": 1e-3,
        "epochs_stage1": epochs,
        "epochs_stage2": epochs,
        "patience": epochs + 1,
        "pos_weight": 10.0,
        "dice_weight": 0.5,
        "bce_weight": 0.5,
        "mixed_precision": True,
        "use_focal_dice": True,
        "focal_alpha": 0.75,
        "focal_gamma": 2.0,
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        learning_rate=1e-3,
        num_epochs=epochs,
        mixed_precision=True,
        output_dir=str(run_dir),
    )

    trainer.fit(epochs=epochs, patience=epochs + 1)

    hist = trainer.history
    metrics = {
        "tumor_weight": tumor_weight,
        "train_loss": float(hist["train_loss"][-1]) if hist.get("train_loss") else None,
        "val_dice": float(hist["val_dice"][-1]) if hist.get("val_dice") else None,
        "val_tumor_dice": float(hist["val_tumor_dice"][-1]) if hist.get("val_tumor_dice") else None,
        "val_auprc": float(hist["val_auprc"][-1]) if hist.get("val_auprc") else None,
        "val_auroc": float(hist["val_auroc"][-1]) if hist.get("val_auroc") else None,
        "val_fg_frac": float(hist["val_fg_frac"][-1]) if hist.get("val_fg_frac") else None,
    }

    # Also compute raw-logit AUPRC (not thresholded) as an extra diagnostic
    raw_auprc = compute_val_auprc(model, val_loader)
    metrics["raw_val_auprc"] = raw_auprc

    # Precision/Recall from last validation batch
    from src.metrics import precision_recall
    model.eval()
    device = DEVICE
    all_preds, all_masks = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = torch.sigmoid(model(images))
            pred_binary = (preds > 0.5).float()
            all_preds.append(pred_binary.cpu())
            all_masks.append(masks.cpu())
    all_preds_t = torch.cat(all_preds, dim=0)
    all_masks_t = torch.cat(all_masks, dim=0)
    pr, re = precision_recall(all_preds_t, all_masks_t)
    metrics["precision"] = float(pr.mean().item())
    metrics["recall"] = float(re.mean().item())

    # Save run metadata
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  [sweep] w={tumor_weight}: Dice={metrics['val_dice']:.4f}, "
          f"T Dice={metrics['val_tumor_dice']:.4f}, AUPRC={metrics['val_auprc']:.4f}, "
          f"FG%={metrics['val_fg_frac']*100:.2f}%, Prec={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}")

    return metrics


def aggregate_sweep_results(results: List[dict], output_dir: str = "experiments/sprint1") -> pd.DataFrame:
    """One row per config, saved to sweep_results.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df = df.sort_values("tumor_weight").reset_index(drop=True)
    csv_path = output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[aggregate] Saved {len(df)} configs to {csv_path}")

    best = select_best_config(df)
    print(f"\n[selection] {best}")
    return df


def select_best_config(df: pd.DataFrame) -> dict:
    """
    Rank configs by a combined score, then flag metric disagreements.
    Uses z-score sum of: T Dice, AUPRC, Precision, minus z-score of (1 - Dice)
    to penalize overall Dice loss. Prints warnings when metrics disagree.
    """
    from scipy.stats import zscore

    cols = ["val_tumor_dice", "val_auprc", "precision", "val_dice"]
    available = [c for c in cols if c in df.columns and df[c].nunique() > 1]

    scores = pd.Series(0.0, index=df.index)
    details = {}

    for col in available:
        z = zscore(df[col].values, ddof=0)
        sign = -1 if col == "val_dice" else 1  # Dice can conflict with T Dice
        if col == "val_dice":
            scores -= z  # penalize sacrificing overall Dice too much
        else:
            scores += z
        details[col] = z

    scores -= df["tumor_weight"].rank() * 0.1  # mild penalty for higher weight

    best_idx = scores.idxmax()
    best = df.loc[best_idx].to_dict()

    # Flag disagreements
    auprc_best = df.loc[df["val_auprc"].idxmax(), "tumor_weight"]
    tdice_best = df.loc[df["val_tumor_dice"].idxmax(), "tumor_weight"]
    prec_best = df.loc[df["precision"].idxmax(), "tumor_weight"]

    best["auprc_best_weight"] = auprc_best
    best["tdice_best_weight"] = tdice_best
    best["precision_best_weight"] = prec_best

    winners = {auprc_best, tdice_best, prec_best}
    if len(winners) > 1:
        print(f"  ⚠  Metric disagreement: AUPRC favors w={auprc_best:.0f}, "
              f"T Dice favors w={tdice_best:.0f}, Precision favors w={prec_best:.0f}")
        print(f"  → Combined score picks w={best['tumor_weight']:.0f}")

    # Check precision/recall trajectory (v2 collapse detection)
    sorted_df = df.sort_values("tumor_weight")
    pr_trend = sorted_df["precision"].diff().dropna()
    if (pr_trend < 0).all():
        print(f"  ⚠  Precision declines monotonically with weight — "
              f"higher weights approach v2-style collapse (Prec→0, Rec→1)")
        print(f"  → Preferring lower weight when tied")

    return best

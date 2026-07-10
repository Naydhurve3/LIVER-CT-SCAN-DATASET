import argparse
import gc
import importlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.framework.core.reproducibility import set_seed
from src.framework.core.factory import build_model, build_loss
from src.framework.data.lits_dataset import DataPathManager, VolumeWiseSplitter, create_2d_dataloaders
from src.framework.data.transforms import PreprocessingTransform
from src.framework.evaluation.metrics import dice_coefficient, iou_score
from src.framework.evaluation.surface_metrics import compute_all_surface_metrics
from src.framework.training.trainer import Trainer
from src.framework.utils.gpu_utils import DEVICE
from src.framework.utils.logging_utils import setup_logging

LOGGER = setup_logging(__name__)


BASE_CONFIG = {
    "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
    "loss": {"name": "combined", "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
    "training": {"epochs": 5, "batch_size": 8, "mixed_precision": True, "lr": 1e-3},
}


ABLATIONS = [
    {
        "name": "baseline",
        "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
        "loss": {"name": "combined", "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
        "ensemble": 1,
    },
    {
        "name": "baseline_uwaclv1",
        "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
        "loss": {"name": "uwacl_v1", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
        "ensemble": 1,
    },
    {
        "name": "baseline_uwaclv2",
        "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
        "loss": {"name": "uwacl_v2", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0, "edge_weight": 0.1},
        "ensemble": 1,
    },
    {
        "name": "ensemble3",
        "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
        "loss": {"name": "combined", "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
        "ensemble": 3,
    },
    {
        "name": "ensemble3_uwaclv1",
        "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
        "loss": {"name": "uwacl_v1", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
        "ensemble": 3,
    },
    {
        "name": "ensemble3_uwaclv2",
        "model": {"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1},
        "loss": {"name": "uwacl_v2", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0, "edge_weight": 0.1},
        "ensemble": 3,
    },
    {
        "name": "faupnet",
        "model": {"name": "faupnet", "in_channels": 1, "out_channels": 1, "gate_levels": [3, 4]},
        "loss": {"name": "combined", "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
        "ensemble": 1,
    },
    {
        "name": "faupnet_uwaclv1",
        "model": {"name": "faupnet", "in_channels": 1, "out_channels": 1, "gate_levels": [3, 4]},
        "loss": {"name": "uwacl_v1", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0},
        "ensemble": 1,
    },
    {
        "name": "faupnet_uwaclv2",
        "model": {"name": "faupnet", "in_channels": 1, "out_channels": 1, "gate_levels": [3, 4]},
        "loss": {"name": "uwacl_v2", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0, "edge_weight": 0.1},
        "ensemble": 1,
    },
    {
        "name": "faupnet_ensemble3_uwaclv2",
        "model": {"name": "faupnet", "in_channels": 1, "out_channels": 1, "gate_levels": [3, 4]},
        "loss": {"name": "uwacl_v2", "beta": 5.0, "tau": 0.1, "dice_weight": 0.5, "bce_weight": 0.5, "pos_weight": 10.0, "edge_weight": 0.1},
        "ensemble": 3,
    },
]


def _get_val_loader():
    path_manager = DataPathManager(
        images_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset",
        masks_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks",
    )
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(Path("data/splits"))
    val_vids = splits.get("val", [])[:1]
    transform = PreprocessingTransform(target_size=(256, 256), hu_low=-100, hu_high=400)
    _, val_loader, _ = create_2d_dataloaders(
        volume_index, [], val_vids, [],
        batch_size=4, num_workers=0, transform_val=transform,
    )
    return val_loader


def _get_train_val_loaders():
    path_manager = DataPathManager(
        images_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset",
        masks_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks",
    )
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(Path("data/splits"))
    val_vids = splits.get("val", [])[:1]
    train_vids = splits.get("train", [])[:2]
    transform = PreprocessingTransform(target_size=(256, 256), hu_low=-100, hu_high=400)
    _, val_loader, _ = create_2d_dataloaders(
        volume_index, [], val_vids, [],
        batch_size=4, num_workers=0, transform_val=transform,
    )
    train_loader, _, _ = create_2d_dataloaders(
        volume_index, train_vids, [], [],
        batch_size=4, num_workers=0, transform_train=transform, transform_val=transform,
    )
    return train_loader, val_loader


def evaluate_model(model, val_loader):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            targets = batch["mask"].to(DEVICE)
            preds = model(images)
            pred_binary = (preds > 0.5).float()
            for i in range(pred_binary.shape[0]):
                d = dice_coefficient(pred_binary[i:i+1], targets[i:i+1]).item()
                iou = iou_score(pred_binary[i:i+1], targets[i:i+1]).item()
                dices.append(d)
                ious.append(iou)
    return {"dice": float(np.mean(dices)), "dice_std": float(np.std(dices)),
            "iou": float(np.mean(ious)), "iou_std": float(np.std(ious))}


def run_single_ablation(cfg):
    set_seed(42)
    model = build_model(cfg["model"])
    model = model.to(DEVICE)
    val_loader = _get_val_loader()
    result = evaluate_model(model, val_loader)
    del val_loader
    result.update({"model": cfg["model"]["name"], "loss": cfg["loss"]["name"], "ensemble": cfg.get("ensemble", 1)})
    return result


def run_training_ablation(cfg, epochs=3):
    set_seed(42)
    model = build_model(cfg["model"])
    model = model.to(DEVICE)
    criterion = build_loss(cfg["loss"])
    train_loader, val_loader = _get_train_val_loaders()
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      num_epochs=epochs, mixed_precision=True)
    trainer.criterion = criterion
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    result = evaluate_model(model, val_loader)
    result["train_time"] = round(train_time, 2)
    del train_loader, val_loader
    result.update({"model": cfg["model"]["name"], "loss": cfg["loss"]["name"], "ensemble": cfg.get("ensemble", 1)})
    return result


def run_ablation_suite(mode="zero_shot", epochs=3, output_dir="research/ablation"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in ABLATIONS:
        name = cfg["name"]
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"  Model: {cfg['model']['name']}, Loss: {cfg['loss']['name']}, Ensemble: {cfg['ensemble']}")
        try:
            if mode == "zero_shot":
                result = run_single_ablation(cfg)
            else:
                result = run_training_ablation(cfg, epochs)
            result["status"] = "ok"
            print(f"  Dice: {result['dice']:.4f} +/- {result['dice_std']:.4f}")
        except Exception as e:
            result = {"name": name, "status": "error", "error": str(e)}
            print(f"  ERROR: {e}")
        result["name"] = name
        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    table_path = output_dir / "ablation_results.json"
    table_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*50}")
    print(f"Results saved to {table_path}")
    print(f"{'='*50}")
    print(f"{'Experiment':<30} {'Dice':<10} {'IoU':<10} {'Status':<10}")
    print("-" * 60)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['name']:<30} {r['dice']:.4f}    {r['iou']:.4f}    {r['status']:<10}")
        else:
            print(f"{r['name']:<30} {'N/A':<10} {'N/A':<10} {r['status']:<10}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation runner for FAUP-Net experiments")
    parser.add_argument("--mode", choices=["zero_shot", "train"], default="zero_shot",
                        help="zero_shot: evaluate untrained models; train: train each config")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per ablation (train mode)")
    parser.add_argument("--output", type=str, default="research/ablation", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List all ablation experiments")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':<30} {'Model':<20} {'Loss':<20} {'Ensemble':<10}")
        print("-" * 80)
        for cfg in ABLATIONS:
            print(f"{cfg['name']:<30} {cfg['model']['name']:<20} {cfg['loss']['name']:<20} {cfg['ensemble']:<10}")
        return

    run_ablation_suite(mode=args.mode, epochs=args.epochs, output_dir=args.output)


if __name__ == "__main__":
    main()

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.framework.core.config import build_experiment_config
from src.framework.core.factory import build_model
from src.framework.evaluation.metrics import dice_coefficient, iou_score
from src.framework.training.trainer import Trainer


def kfold_split(volume_ids, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    ids = sorted(volume_ids)
    rng.shuffle(ids)
    folds = []
    fold_size = len(ids) // n_folds
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(ids)
        val_ids = ids[start:end]
        train_ids = [v for v in ids if v not in val_ids]
        folds.append({"train": train_ids, "val": val_ids})
    return folds


class CrossValidator:
    def __init__(
        self,
        config_path: str,
        build_trainer_fn: Optional[Callable] = None,
        output_dir: str = "experiments/cv",
        device: str = "cpu",
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.build_trainer_fn = build_trainer_fn or self._default_build_trainer
        self.cfg = build_experiment_config(config_path)

    def _default_build_trainer(self, fold_cfg, fold, device):
        model = build_model(fold_cfg.get("model", {}))
        return Trainer(model=model, config=fold_cfg, device=device, run_name=f"fold_{fold}")

    def _score_model(self, model, val_loader):
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                targets = batch["mask"].to(self.device)
                preds = model(images)
                pred_binary = (preds > 0.5).float()
                dices.append(dice_coefficient(pred_binary, targets).item())
                ious.append(iou_score(pred_binary, targets).item())
        return {
            "dice": float(np.mean(dices)),
            "dice_std": float(np.std(dices)),
            "iou": float(np.mean(ious)),
            "iou_std": float(np.std(ious)),
        }

    def run(self, volume_ids, n_folds=5, dataloader_fn=None):
        if dataloader_fn is None:
            raise ValueError("dataloader_fn required: callable(fold_train_ids, fold_val_ids) -> (train_loader, val_loader)")
        folds = kfold_split(volume_ids, n_folds=n_folds)
        results = []
        for fold_idx, fold in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
            train_loader, val_loader = dataloader_fn(fold["train"], fold["val"])
            trainer = self.build_trainer_fn(self.cfg, fold_idx, self.device)
            trained_model = trainer.train(train_loader, val_loader)
            fold_result = self._score_model(trained_model, val_loader)
            fold_result["fold"] = fold_idx
            fold_result["train_volumes"] = fold["train"]
            fold_result["val_volumes"] = fold["val"]
            results.append(fold_result)
            (self.output_dir / f"fold_{fold_idx}_results.json").write_text(
                json.dumps(fold_result, indent=2)
            )
        return self._aggregate(results)

    def _aggregate(self, results):
        dices = [r["dice"] for r in results]
        ious = [r["iou"] for r in results]
        return {
            "n_folds": len(results),
            "dice_mean": float(np.mean(dices)),
            "dice_std": float(np.std(dices, ddof=1)),
            "dice_min": float(np.min(dices)),
            "dice_max": float(np.max(dices)),
            "iou_mean": float(np.mean(ious)),
            "iou_std": float(np.std(ious, ddof=1)),
            "per_fold": results,
        }

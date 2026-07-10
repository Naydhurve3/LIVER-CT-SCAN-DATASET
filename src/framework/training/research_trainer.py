from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.framework.evaluation.checkpoints import write_json
from src.framework.evaluation.checkpoints import load_checkpoint_into_model
from src.framework.evaluation.research_metrics import select_threshold


def aggregate_uncertainty_maps(model: torch.nn.Module, output_size) -> Optional[torch.Tensor]:
    if not hasattr(model, "get_uncertainty_maps"):
        return None
    maps = list(model.get_uncertainty_maps().values())
    if not maps:
        return None
    resized = [F.interpolate(value, size=output_size, mode="bilinear", align_corners=False)
               for value in maps]
    return torch.stack(resized).mean(dim=0)


class ResearchTrainer:
    """Deterministic trainer selecting checkpoints by positive-volume Dice."""

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 train_loader, val_loader, output_dir: str | Path,
                 epochs: int = 10, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, mixed_precision: bool = True,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.history = []
        self.best_score = float("-inf")
        self.best_epoch = 0

    def _loss(self, logits, targets):
        uncertainty = aggregate_uncertainty_maps(self.model, logits.shape[-2:])
        if uncertainty is not None:
            try:
                return self.loss_fn(logits, targets, uncertainty)
            except TypeError:
                pass
        return self.loss_fn(logits, targets)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        if hasattr(self.loss_fn, "set_epoch"):
            self.loss_fn.set_epoch(epoch, self.epochs)
        total = 0.0
        steps = 0
        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["mask"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self._loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1)

    def _save(self, path: Path, epoch: int) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch, "best_positive_volume_dice": self.best_score,
            "history": self.history,
        }, path)

    def fit(self) -> Dict[str, object]:
        started = time.perf_counter()
        for epoch in range(self.epochs):
            loss = self.train_epoch(epoch)
            validation = select_threshold(self.model, self.val_loader, self.device, [0.5])
            score = validation["ranking"][0]["positive_volume_dice"]
            self.scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": loss,
                   "val_positive_volume_dice_at_0_5": score,
                   "lr": self.optimizer.param_groups[0]["lr"]}
            self.history.append(row)
            print(
                f"epoch={epoch + 1}/{self.epochs} loss={loss:.6f} "
                f"val_positive_volume_dice={score:.4f}",
                flush=True,
            )
            self._save(self.output_dir / "last_checkpoint.pth", epoch + 1)
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch + 1
                self._save(self.output_dir / "best_checkpoint.pth", epoch + 1)
            write_json(self.output_dir / "history.json", {"epochs": self.history})
        load_checkpoint_into_model(
            self.model, self.output_dir / "best_checkpoint.pth", self.device
        )
        return {
            "best_checkpoint": str(self.output_dir / "best_checkpoint.pth"),
            "best_epoch": self.best_epoch,
            "best_positive_volume_dice": self.best_score,
            "runtime_seconds": time.perf_counter() - started,
            "history": copy.deepcopy(self.history),
        }

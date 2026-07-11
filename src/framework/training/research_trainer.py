from __future__ import annotations

import copy
import math
import random
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.framework.evaluation.checkpoints import load_checkpoint_into_model, write_json
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


def _rng_state() -> Dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Dict[str, object]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["name"], numpy_state["keys"].cpu().numpy().astype(np.uint32),
        numpy_state["position"], numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


class ResearchTrainer:
    """Epoch-resumable trainer selecting checkpoints by positive-volume Dice."""

    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module,
                 train_loader, val_loader, output_dir: str | Path,
                 epochs: int = 10, schedule_epochs: Optional[int] = None,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 mixed_precision: bool = True,
                 device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.schedule_epochs = schedule_epochs or epochs
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.schedule_epochs, 1)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.history = []
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.start_epoch = 0
        self.resume_reproducibility = "new_run"

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
            self.loss_fn.set_epoch(epoch, self.schedule_epochs)
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

    def _sampler_state(self):
        sampler = getattr(self.train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        return generator.get_state() if generator is not None else None

    def _set_sampler_state(self, state) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        if generator is not None and state is not None:
            generator.set_state(state)

    def _checkpoint_payload(self, epoch: int, status: str) -> Dict[str, object]:
        return {
            "checkpoint_version": 2,
            "status": status,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng_state": _rng_state(),
            "sampler_generator_state": self._sampler_state(),
            "epoch": epoch,
            "target_epochs": self.epochs,
            "schedule_epochs": self.schedule_epochs,
            "best_positive_volume_dice": self.best_score,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    def _save(self, path: Path, epoch: int, status: str = "running") -> None:
        torch.save(self._checkpoint_payload(epoch, status), path)

    def resume(self, checkpoint_path: str | Path) -> Dict[str, object]:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location=self.device, weights_only=True)
        stored_schedule_epochs = payload.get("schedule_epochs")
        if stored_schedule_epochs is not None and int(stored_schedule_epochs) != self.schedule_epochs:
            raise ValueError(
                f"Checkpoint schedule horizon is {stored_schedule_epochs}, "
                f"but this run is configured for {self.schedule_epochs}"
            )
        state = payload.get("model_state", payload.get("state_dict", payload))
        self.model.load_state_dict(state)
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        self.history = list(payload.get("history", []))
        self.start_epoch = int(payload.get("epoch", len(self.history)))
        stored_best = float(payload.get("best_positive_volume_dice", float("-inf")))
        history_scores = [float(row["val_positive_volume_dice_at_0_5"])
                          for row in self.history if "val_positive_volume_dice_at_0_5" in row]
        self.best_score = stored_best if math.isfinite(stored_best) else (
            max(history_scores) if history_scores else float("-inf")
        )
        self.best_epoch = int(payload.get("best_epoch", 0))
        if not self.best_epoch and history_scores:
            self.best_epoch = 1 + history_scores.index(max(history_scores))

        if "scheduler_state" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state"])
            self.resume_reproducibility = "exact"
        else:
            base_lrs = [group.get("initial_lr", group["lr"]) for group in self.optimizer.param_groups]
            for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
                group["lr"] = base_lr
                group["initial_lr"] = base_lr
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.schedule_epochs, 1)
            )
            eta_min = self.scheduler.eta_min
            resumed_lrs = [
                eta_min + (base_lr - eta_min) *
                (1.0 + math.cos(math.pi * self.start_epoch / self.schedule_epochs)) / 2.0
                for base_lr in base_lrs
            ]
            for group, resumed_lr in zip(self.optimizer.param_groups, resumed_lrs):
                group["lr"] = resumed_lr
            scheduler_state = self.scheduler.state_dict()
            scheduler_state.update({
                "last_epoch": self.start_epoch,
                "_step_count": self.start_epoch + 1,
                "_last_lr": resumed_lrs,
            })
            self.scheduler.load_state_dict(scheduler_state)
            self.resume_reproducibility = "legacy_scheduler_reconstructed_rng_unavailable"
        if "scaler_state" in payload:
            self.scaler.load_state_dict(payload["scaler_state"])
        if "sampler_generator_state" in payload:
            self._set_sampler_state(payload["sampler_generator_state"])
        elif self.start_epoch:
            for _ in range(self.start_epoch):
                list(iter(self.train_loader.sampler))
        if "rng_state" in payload:
            _restore_rng_state(payload["rng_state"])
        return {
            "path": str(path.resolve()), "start_epoch": self.start_epoch,
            "best_epoch": self.best_epoch, "best_score": self.best_score,
            "resume_reproducibility": self.resume_reproducibility,
        }

    def fit(self) -> Dict[str, object]:
        started = time.perf_counter()
        epoch_start_path = self.output_dir / "epoch_start_checkpoint.pth"
        try:
            for epoch in range(self.start_epoch, self.epochs):
                self._save(epoch_start_path, epoch, status="epoch_start")
                loss = self.train_epoch(epoch)
                validation = select_threshold(self.model, self.val_loader, self.device, [0.5])
                score = validation["ranking"][0]["positive_volume_dice"]
                self.scheduler.step()
                row = {
                    "epoch": epoch + 1, "train_loss": loss,
                    "val_positive_volume_dice_at_0_5": score,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                self.history.append(row)
                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = epoch + 1
                    self._save(self.output_dir / "best_checkpoint.pth", epoch + 1, status="best")
                self._save(self.output_dir / "last_checkpoint.pth", epoch + 1)
                write_json(self.output_dir / "history.json", {"epochs": self.history})
                print(
                    f"epoch={epoch + 1}/{self.epochs} loss={loss:.6f} "
                    f"val_positive_volume_dice={score:.4f}", flush=True,
                )
        except KeyboardInterrupt:
            if epoch_start_path.exists():
                shutil.copy2(epoch_start_path, self.output_dir / "interrupted_checkpoint.pth")
            return {
                "status": "interrupted",
                "resume_checkpoint": str(self.output_dir / "interrupted_checkpoint.pth"),
                "completed_epochs": len(self.history),
                "best_epoch": self.best_epoch,
                "best_positive_volume_dice": self.best_score,
                "runtime_seconds": time.perf_counter() - started,
                "history": copy.deepcopy(self.history),
                "resume_reproducibility": self.resume_reproducibility,
            }
        finally:
            epoch_start_path.unlink(missing_ok=True)

        best_path = self.output_dir / "best_checkpoint.pth"
        if best_path.exists():
            load_checkpoint_into_model(self.model, best_path, self.device)
        return {
            "status": "completed", "best_checkpoint": str(best_path),
            "best_epoch": self.best_epoch,
            "best_positive_volume_dice": self.best_score,
            "completed_epochs": len(self.history),
            "runtime_seconds": time.perf_counter() - started,
            "history": copy.deepcopy(self.history),
            "resume_reproducibility": self.resume_reproducibility,
        }

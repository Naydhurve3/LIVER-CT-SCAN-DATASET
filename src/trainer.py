import os
import json
import copy
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.losses import FocalLoss, FocalDiceLoss, CombinedLoss, UncertaintyWeightedLoss
from src.metrics import dice_coefficient, iou_score, ensemble_uncertainty, calibration_error, compute_all_metrics, slice_level_auprc, slice_level_auroc
from src.utils import print_gpu_memory
from src.models import EnsembleWrapper
from src.config import PHASE4_RESEARCH_CONFIG as CFG
from tqdm import tqdm

warnings.filterwarnings("ignore", message="`torch.cuda.amp.GradScaler`")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast`")


class Trainer:
    def __init__(self, model, train_loader, val_loader, config=None,
                 learning_rate=None, num_epochs=None, mixed_precision=None,
                 output_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir or "models")

        if config is not None:
            self.cfg = dict(config)
        else:
            self.cfg = dict(CFG)
        if learning_rate is not None:
            self.cfg['lr'] = learning_rate
        if num_epochs is not None:
            self.cfg['epochs_stage1'] = num_epochs
            self.cfg['epochs_stage2'] = num_epochs
        if mixed_precision is not None:
            self.cfg['mixed_precision'] = mixed_precision

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FocalDiceLoss = FocalLoss + DiceLoss.
        # Focal handles extreme imbalance, Dice directly penalizes all-bg collapse.
        # Fall back to FocalLoss-only, then CombinedLoss.
        if self.cfg.get('use_focal_dice', False):
            self.criterion = FocalDiceLoss(
                focal_alpha=self.cfg.get('focal_alpha', 0.75),
                focal_gamma=self.cfg.get('focal_gamma', 2.0),
            )
        elif self.cfg.get('use_focal', True):
            self.criterion = FocalLoss(
                alpha=self.cfg.get('focal_alpha', 0.75),
                gamma=self.cfg.get('focal_gamma', 2.0),
            )
        else:
            self.criterion = CombinedLoss(
                dice_weight=self.cfg['dice_weight'],
                bce_weight=self.cfg['bce_weight'],
                pos_weight=self.cfg['pos_weight'],
            )
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=1e-4,
        )
        self.scaler = GradScaler(enabled=self.cfg['mixed_precision'])

        self.best_val_dice = 0.0
        self.best_state = None
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_dice': [], 'val_iou': [],
                        'val_tumor_dice': [], 'val_fg_frac': [], 'val_auprc': [], 'val_auroc': []}

    def train(self, **fit_kwargs):
        return self.fit(**fit_kwargs)

    def train_epoch(self, desc="Train"):
        self.model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(self.train_loader, desc=desc, leave=False,
                    bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg['mixed_precision']):
                preds = self.model(images)
                loss = self.criterion(preds, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            steps += 1
        pbar.close()
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        dices, ious, tumor_dices = [], [], []
        all_probs, all_masks = [], []
        n_pos_pixels, n_total_pixels = 0, 0
        pbar = tqdm(self.val_loader, desc="  Val", leave=False,
                    bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            with autocast(enabled=self.cfg['mixed_precision']):
                preds = torch.sigmoid(self.model(images))
            pred_binary = (preds > 0.5).float()
            dices.append(dice_coefficient(pred_binary, masks).item())
            ious.append(iou_score(pred_binary, masks).item())
            # Tumor-only Dice: only on slices that actually have tumor
            tumor_mask = masks.sum(dim=(1, 2, 3)) > 0.5
            if tumor_mask.any():
                tumor_dices.append(
                    dice_coefficient(pred_binary[tumor_mask], masks[tumor_mask]).item()
                )
            # Accumulate for AUPRC and foreground fraction
            all_probs.append(preds.cpu())
            all_masks.append(masks.cpu())
            n_pos_pixels += pred_binary.sum().item()
            n_total_pixels += pred_binary.numel()
        pbar.close()
        avg_tumor_dice = torch.tensor(tumor_dices).mean().item() if tumor_dices else 0.0
        # Predicted foreground fraction
        fg_frac = n_pos_pixels / max(n_total_pixels, 1)
        # Slice-level AUPRC
        all_probs_t = torch.cat(all_probs, dim=0)
        all_masks_t = torch.cat(all_masks, dim=0)
        auprc = slice_level_auprc(all_probs_t, all_masks_t)
        auroc = slice_level_auroc(all_probs_t, all_masks_t)
        return (torch.tensor(dices).mean().item(), torch.tensor(ious).mean().item(),
                avg_tumor_dice, fg_frac, auprc, auroc)

    def fit(self, epochs=None, patience=None, warmup_epochs=0):
        epochs = epochs or self.cfg.get('epochs_stage1', 25)
        patience = patience or self.cfg.get('patience', 10)
        warmup_epochs = warmup_epochs or self.cfg.get('warmup_epochs', 0)
        ckpt_interval = max(1, epochs // 10)
        self.warmup_epochs = warmup_epochs
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return (epoch + 1) / warmup_epochs
                return 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs - warmup_epochs
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
            self.main_scheduler = None
        self.best_val_dice = 0.0
        self.best_state = None
        self.patience_counter = 0
        start_time = time.time()
        n_train = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 0
        n_val = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 0
        print(f"\n{'='*60}")
        print(f"  Training: {n_train:,} slices | Val: {n_val:,} slices | Epochs: {epochs}")
        print(f"  Output: {self.output_dir}/")
        print(f"{'='*60}")
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            print_gpu_memory()
            train_loss = self.train_epoch(desc=f"  Train")
            (val_dice, val_iou, val_tumor_dice, val_fg_frac, val_auprc, val_auroc) = self.validate()
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            if hasattr(self, 'main_scheduler') and self.main_scheduler is not None and epoch >= warmup_epochs - 1:
                self.main_scheduler.step()
            self.history['train_loss'].append(train_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            self.history['val_tumor_dice'].append(val_tumor_dice)
            self.history['val_fg_frac'].append(val_fg_frac)
            self.history['val_auprc'].append(val_auprc)
            self.history['val_auroc'].append(val_auroc)
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            marker = " *BEST*" if val_dice > self.best_val_dice else ""
            print(f"  Loss:{train_loss:.4f} | Dice:{val_dice:.4f} | IoU:{val_iou:.4f} | T Dice:{val_tumor_dice:.4f} | FG:{val_fg_frac:.4f} | AUPRC:{val_auprc:.4f}")
            print(f"  Time: {epoch_time:.0f}s epoch | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining{marker}")
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                ckpt_path = self.output_dir / "best_model.pth"
                self.save_checkpoint(ckpt_path)
                print(f"  -> Saved best model (dice={val_dice:.4f}) to {ckpt_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
            if (epoch + 1) % ckpt_interval == 0:
                ckpt_path = self.output_dir / f"checkpoint_epoch{epoch+1}.pth"
                self.save_checkpoint(ckpt_path)
                print(f"  -> Saved periodic checkpoint to {ckpt_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.output_dir / "final_model.pth"
        self.save_checkpoint(final_path)
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Best val dice: {self.best_val_dice:.4f}")
        print(f"  Saved: {self.output_dir}/final_model.pth")
        print(f"  Saved: {self.output_dir}/best_model.pth")
        print(f"{'='*60}")
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.model

    def evaluate(self, test_loader):
        result = evaluate_model(self.model, test_loader, self.device)
        return result

    def save_checkpoint(self, path):
        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'history': self.history,
        }, path)

    @staticmethod
    def load_checkpoint(path, model, optimizer=None, map_location='cpu'):
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        history = checkpoint.get('history', {})
        print(f"  Loaded checkpoint: {path}")
        print(f"  Best val dice: {best_val_dice:.4f}")
        return model, optimizer, best_val_dice, history


class UncertaintyPrecomputer:
    def __init__(self, ensemble_models, train_loader, device, output_dir='tmp/uncertainty'):
        self.ensemble = EnsembleWrapper(ensemble_models)
        self.train_loader = train_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        num_batches = len(self.train_loader)
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            mean_pred, variance = self.ensemble.predict_with_uncertainty(
                images, self.device, sequential=True
            )
            torch.save({
                'uncertainty': variance.float(),
                'mean': mean_pred.float(),
            }, self.output_dir / f'batch_{batch_idx:05d}.pt')
            if batch_idx % 500 == 0:
                print(f"  Uncertainty pre-compute: {batch_idx}/{num_batches}")
        return num_batches


class UncertaintyGuidedTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, config=None):
        super().__init__(model, train_loader, val_loader, config)
        self.criterion = UncertaintyWeightedLoss(
            beta=self.cfg.get('uwacl_beta', 5.0),
            tau=self.cfg.get('uwacl_tau', 0.1),
            dice_weight=self.cfg['dice_weight'],
            bce_weight=self.cfg['bce_weight'],
            pos_weight=self.cfg['pos_weight'],
        )
        self.uncertainty_dir = Path('tmp/uncertainty')

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        steps = 0
        progress = epoch / max(total_epochs - 1, 1)
        tau = 0.1 * max(0.01, (1.0 - 0.9 * progress))
        self.criterion.set_tau(tau)
        pbar = tqdm(self.train_loader, desc=f"  Train UW", leave=False,
                    bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            uncertainty = batch.get('uncertainty')
            if uncertainty is not None:
                uncertainty = uncertainty.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.cfg['mixed_precision']):
                preds = self.model(images)
                loss = self.criterion(preds, masks, uncertainty)
            if torch.isnan(loss) or torch.isinf(loss):
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                if mem > 3.5:
                    torch.cuda.empty_cache()
                continue
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            steps += 1
        pbar.close()
        return total_loss / max(steps, 1)

    def fit(self, epochs=None, patience=None):
        epochs = epochs or self.cfg.get('epochs_stage2', 25)
        patience = patience or self.cfg.get('patience', 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.best_val_dice = 0.0
        self.patience_counter = 0
        start_time = time.time()
        n_train = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 0
        n_val = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 0
        print(f"\n{'='*60}")
        print(f"  Uncertainty-Guided Training: {n_train:,} slices | Val: {n_val:,} | Epochs: {epochs}")
        print(f"{'='*60}")
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            print_gpu_memory()
            train_loss = self.train_epoch(epoch, epochs)
            (val_dice, val_iou, val_tumor_dice, val_fg_frac, val_auprc, val_auroc) = self.validate()
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            self.history['train_loss'].append(train_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            self.history['val_tumor_dice'].append(val_tumor_dice)
            self.history['val_fg_frac'].append(val_fg_frac)
            self.history['val_auprc'].append(val_auprc)
            self.history['val_auroc'].append(val_auroc)
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
            marker = " *BEST*" if val_dice > self.best_val_dice else ""
            print(f"  Loss:{train_loss:.4f} | Dice:{val_dice:.4f} | IoU:{val_iou:.4f} | T Dice:{val_tumor_dice:.4f} | FG:{val_fg_frac:.4f} | AUPRC:{val_auprc:.4f}")
            print(f"  Time: {epoch_time:.0f}s epoch | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining{marker}")
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                ckpt_path = self.output_dir / "best_model_uw.pth"
                self.save_checkpoint(ckpt_path)
                print(f"  -> Saved best model (dice={val_dice:.4f}) to {ckpt_path}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Best val dice: {self.best_val_dice:.4f}")
        print(f"{'='*60}")
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.model


def evaluate_model(model, test_loader, device, return_uncertainty=False):
    model.eval()
    # Accumulate across all batches
    all_preds, all_masks, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            if hasattr(model, 'predict_with_uncertainty'):
                mean_pred, variance = model.predict_with_uncertainty(images, device)
                pred_binary = (mean_pred > 0.5).float()
                all_probs.append(mean_pred.cpu())
                if return_uncertainty:
                    all_preds.append(variance.cpu())
            else:
                preds = torch.sigmoid(model(images))
                pred_binary = (preds > 0.5).float()
                all_probs.append(preds.cpu())
            all_preds.append(pred_binary.cpu())
            all_masks.append(masks.cpu())
    # Stack and compute comprehensive metrics
    all_preds_t = torch.cat(all_preds, dim=0)
    all_masks_t = torch.cat(all_masks, dim=0)
    all_probs_t = torch.cat(all_probs, dim=0) if all_probs else None
    result = compute_all_metrics(all_preds_t, all_masks_t, probs=all_probs_t)
    return result

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    print(f"  Classes: Trainer, UncertaintyPrecomputer, UncertaintyGuidedTrainer")
    print(f"  Functions: evaluate_model")
    print("  OK (requires dataloaders for full test)")

from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.framework.losses.combined_loss import CombinedLoss
from src.framework.losses.uwacl_v1 import UncertaintyWeightedLoss
from src.framework.evaluation.metrics import dice_coefficient, iou_score, calibration_error
from src.framework.uncertainty.ensemble import EnsembleWrapper


class Trainer:
    def __init__(self, model, train_loader, val_loader, config=None,
                 learning_rate=None, num_epochs=None, mixed_precision=None,
                 output_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir or "models")

        self.cfg = dict(config) if config is not None else {}
        if learning_rate is not None:
            self.cfg['lr'] = learning_rate
        if num_epochs is not None:
            self.cfg['epochs'] = num_epochs
        if mixed_precision is not None:
            self.cfg['mixed_precision'] = mixed_precision

        self.cfg.setdefault('lr', 1e-3)
        self.cfg.setdefault('epochs', 50)
        self.cfg.setdefault('mixed_precision', True)
        self.cfg.setdefault('dice_weight', 0.5)
        self.cfg.setdefault('bce_weight', 0.5)
        self.cfg.setdefault('pos_weight', 10.0)
        self.cfg.setdefault('patience', 10)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        self.history = {'train_loss': [], 'val_dice': [], 'val_iou': []}

    def train(self):
        return self.fit()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        steps = 0
        for batch in self.train_loader:
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
        return total_loss / max(steps, 1)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        dices, ious = [], []
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            with autocast(enabled=self.cfg['mixed_precision']):
                preds = torch.sigmoid(self.model(images))
            pred_binary = (preds > 0.5).float()
            dices.append(dice_coefficient(pred_binary, masks).item())
            ious.append(iou_score(pred_binary, masks).item())
        return torch.tensor(dices).mean().item(), torch.tensor(ious).mean().item()

    def fit(self, epochs=None, patience=None):
        epochs = epochs or self.cfg.get('epochs', 50)
        patience = patience or self.cfg.get('patience', 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.best_val_dice = 0.0
        self.patience_counter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_dice, val_iou = self.validate()
            self.scheduler.step()
            self.history['train_loss'].append(train_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.model

    def evaluate(self, test_loader):
        return evaluate_model(self.model, test_loader, self.device)

    def save_checkpoint(self, path):
        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'history': self.history,
        }, path)


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
        for batch_idx, batch in enumerate(self.train_loader):
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
        return total_loss / max(steps, 1)

    def fit(self, epochs=None, patience=None):
        epochs = epochs or self.cfg.get('epochs_stage2', 25)
        patience = patience or self.cfg.get('patience', 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.best_val_dice = 0.0
        self.patience_counter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch, epochs)
            val_dice, val_iou = self.validate()
            self.scheduler.step()
            self.history['train_loss'].append(train_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.model


def evaluate_model(model, test_loader, device, return_uncertainty=False):
    model.eval()
    dices, ious, eces = [], [], []
    all_vars = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            if hasattr(model, 'predict_with_uncertainty'):
                mean_pred, variance = model.predict_with_uncertainty(images, device)
                pred_binary = (mean_pred > 0.5).float()
                all_vars.append(variance)
                probs = mean_pred
            else:
                preds = torch.sigmoid(model(images))
                pred_binary = (preds > 0.5).float()
                probs = preds
            dices.append(dice_coefficient(pred_binary, masks).item())
            ious.append(iou_score(pred_binary, masks).item())
            if probs.numel() > 1000:
                ece = calibration_error(probs.cpu(), masks.cpu())
                eces.append(ece)
    result = {
        'dice': torch.tensor(dices).mean().item(),
        'iou': torch.tensor(ious).mean().item(),
        'ece': torch.tensor(eces).mean().item() if eces else 0.0,
    }
    if return_uncertainty and all_vars:
        result['uncertainty_mean'] = torch.cat(all_vars).mean().item()
    return result

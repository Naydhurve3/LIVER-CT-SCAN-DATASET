from pathlib import Path

import torch

from src.framework.core.registry import MODELS
from src.framework.evaluation.metrics import ensemble_uncertainty


class EnsembleWrapper:
    def __init__(self, models):
        self.models = models

    def to(self, device):
        for m in self.models:
            m.to(device)
        return self

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def predict_with_uncertainty(self, x, device, sequential=True):
        preds = []
        if sequential:
            for m in self.models:
                m = m.to(device)
                m.eval()
                with torch.no_grad():
                    p = torch.sigmoid(m(x))
                preds.append(p.cpu())
                m.to('cpu')
                torch.cuda.empty_cache()
        else:
            for m in self.models:
                m.eval()
                with torch.no_grad():
                    p = torch.sigmoid(m(x.to(device)))
                preds.append(p)
        stacked = torch.stack(preds, dim=0)
        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0)
        return mean_pred, variance


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
        return num_batches

"""1-epoch sweep across tumor_weights 2, 3, 5 to pick the right balance."""
import sys, json, time, multiprocessing, os
os.environ["PYARROW_IGNORE_ZERO_COPY"] = "1"
# Pre-import sklearn before any project code to avoid pyarrow Windows TxF issue
import sklearn.metrics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.config import DEVICE
from src.data_loader import (DatasetConfig, DataPathManager, VolumeWiseSplitter,
                              create_2d_dataloaders)
from src.preprocessing import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.models import create_model, count_params
from src.trainer import Trainer
from src.config import PHASE4_RESEARCH_CONFIG
from src.utils import set_seed

WEIGHTS_TO_TEST = [2, 3, 5]
BATCH_SIZE = 4
LR = 1e-3
SEED = 42
EPOCHS = 1


def main():
    print("=" * 70)
    print("Sampler weight sweep: 1 epoch each")
    print("=" * 70)

    set_seed(SEED)
    path_manager = DataPathManager()
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(DatasetConfig.SPLITS_DIR)

    clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
    transform_train = AugmentedPreprocessingTransform((256, 256), -100, 400, clahe)
    transform_val = PreprocessingTransform((256, 256), -100, 400)

    results = []
    for wt in WEIGHTS_TO_TEST:
        print(f"\n{'='*70}")
        print(f"  weight={wt}")
        print(f"{'='*70}")
        set_seed(SEED)

        train_loader, val_loader, test_loader = create_2d_dataloaders(
            volume_index, splits['train'], splits['val'], splits['test'],
            batch_size=BATCH_SIZE, num_workers=0,
            transform_train=transform_train, transform_val=transform_val,
            use_tumor_sampler=True, tumor_sampler_weight=wt)

        model = create_model('mobilenetv2_unet', in_channels=1, out_channels=1, pretrained=True).to(DEVICE)
        train_config = dict(PHASE4_RESEARCH_CONFIG)
        train_config['use_focal_dice'] = True
        train_config['focal_alpha'] = 0.75

        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                          config=train_config, learning_rate=LR, num_epochs=EPOCHS,
                          mixed_precision=True, output_dir=f"models/sweep_w{wt}")

        trainer.fit(epochs=EPOCHS, patience=100)

        h = trainer.history
        row = {
            'weight': wt,
            'train_loss': h['train_loss'][-1] if h.get('train_loss') else -1,
            'val_dice': h['val_dice'][-1] if h.get('val_dice') else -1,
            'val_tumor_dice': h['val_tumor_dice'][-1] if h.get('val_tumor_dice') else -1,
            'fg_frac': h['val_fg_frac'][-1] if h.get('val_fg_frac') else -1,
            'auprc': h['val_auprc'][-1] if h.get('val_auprc') else -1,
            'auroc': h['val_auroc'][-1] if h.get('val_auroc') else -1,
        }
        results.append(row)
        print(f"  => loss={row['train_loss']:.4f} dice={row['val_dice']:.4f} "
              f"tumor_dice={row['val_tumor_dice']:.4f} fg={row['fg_frac']:.4f} "
              f"auprc={row['auprc']:.4f} auroc={row['auroc']:.4f}")

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'w':>6}  {'Loss':>8}  {'Dice':>8}  {'T_Dice':>8}  {'FG%':>8}  {'AUPRC':>8}  {'AUROC':>8}")
    print("-" * 70)
    best_auprc, best_w = -1, None
    for r in results:
        fg_pct = r['fg_frac'] * 100 if r['fg_frac'] >= 0 else -1
        print(f"{r['weight']:>6}  {r['train_loss']:>8.4f}  {r['val_dice']:>8.4f}  {r['val_tumor_dice']:>8.4f}  {fg_pct:>7.2f}%  {r['auprc']:>8.4f}  {r['auroc']:>8.4f}")
        if r['auprc'] > best_auprc:
            best_auprc = r['auprc']
            best_w = r['weight']

    print("-" * 70)
    print(f"Best weight: {best_w} (highest AUPRC={best_auprc:.4f})")

    out = Path("models/sweep_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved to {out}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

"""5-epoch pilot at tumor_weight=3 (best from sweep)."""
import sys, json, os
os.environ["PYARROW_IGNORE_ZERO_COPY"] = "1"
import sklearn.metrics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import DEVICE, PHASE4_RESEARCH_CONFIG
from src.data_loader import DatasetConfig, DataPathManager, VolumeWiseSplitter, create_2d_dataloaders
from src.preprocessing import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.models import create_model, count_params
from src.trainer import Trainer
from src.utils import set_seed

EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-3
SEED = 42
WEIGHT = 3
OUTPUT_DIR = Path("models/s9_pilot_v3")


def main():
    print("=" * 60)
    print(f"Pilot v3: {EPOCHS} epochs, tumor_weight={WEIGHT}")
    print("=" * 60)
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path_manager = DataPathManager()
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(DatasetConfig.SPLITS_DIR)

    clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
    transform_train = AugmentedPreprocessingTransform((256, 256), -100, 400, clahe)
    transform_val = PreprocessingTransform((256, 256), -100, 400)

    train_loader, val_loader, test_loader = create_2d_dataloaders(
        volume_index, splits['train'], splits['val'], splits['test'],
        batch_size=BATCH_SIZE, num_workers=0,
        transform_train=transform_train, transform_val=transform_val,
        use_tumor_sampler=True, tumor_sampler_weight=WEIGHT)

    model = create_model('mobilenetv2_unet', in_channels=1, out_channels=1, pretrained=True).to(DEVICE)
    print(f"Parameters: {count_params(model):,}")

    train_config = dict(PHASE4_RESEARCH_CONFIG)
    train_config['use_focal_dice'] = True
    train_config['focal_alpha'] = 0.75

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      config=train_config, learning_rate=LR, num_epochs=EPOCHS,
                      mixed_precision=True, output_dir=str(OUTPUT_DIR))

    trainer.fit(epochs=EPOCHS, patience=10)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_metrics = trainer.evaluate(test_loader)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(OUTPUT_DIR / 'history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in trainer.history.items()}, f, indent=2)
    with open(OUTPUT_DIR / 'test_metrics.json', 'w') as f:
        tm = {}
        for k, v in test_metrics.items():
            if isinstance(v, torch.Tensor):
                tm[k] = float(v.item()) if v.numel() == 1 else v.tolist()
            else:
                tm[k] = float(v)
        json.dump(tm, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

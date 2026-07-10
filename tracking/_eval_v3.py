"""Load saved s9_pilot_v3 model and run test evaluation only (no training)."""
import sys, json, os
os.environ["PYARROW_IGNORE_ZERO_COPY"] = "1"
import sklearn.metrics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.config import DEVICE, PHASE4_RESEARCH_CONFIG
from src.data_loader import DatasetConfig, DataPathManager, VolumeWiseSplitter, create_2d_dataloaders
from src.preprocessing import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.models import create_model
from src.trainer import Trainer

BATCH_SIZE = 4
SEED = 42
MODEL_DIR = Path(__file__).resolve().parent.parent / "models/s9_finetune_v4"


def main():
    print("=" * 60)
    print("Evaluating s9_pilot_v3 on test set (no training)")
    print("=" * 60)

    path_manager = DataPathManager()
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(DatasetConfig.SPLITS_DIR)

    clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
    transform_val = PreprocessingTransform((256, 256), -100, 400)

    _, _, test_loader = create_2d_dataloaders(
        volume_index, splits['train'], splits['val'], splits['test'],
        batch_size=BATCH_SIZE, num_workers=0,
        transform_train=transform_val, transform_val=transform_val,
        use_tumor_sampler=False)

    model = create_model('mobilenetv2_unet', in_channels=1, out_channels=1, pretrained=True).to(DEVICE)

    # Load best checkpoint
    best_path = MODEL_DIR / "best_model.pth"
    final_path = MODEL_DIR / "final_model.pth"
    ckpt_path = best_path if best_path.exists() else final_path
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model_state'])
    print(f"Loaded model from {ckpt_path}")

    # Minimal trainer just for evaluate()
    train_config = dict(PHASE4_RESEARCH_CONFIG)
    train_config['use_focal_dice'] = True
    train_config['focal_alpha'] = 0.75

    trainer = Trainer(model=model, train_loader=None, val_loader=None,
                      config=train_config, learning_rate=1e-3, num_epochs=1,
                      mixed_precision=True, output_dir=str(MODEL_DIR))

    print("\nRunning test evaluation...")
    test_metrics = trainer.evaluate(test_loader)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    with open(MODEL_DIR / 'test_metrics.json', 'w') as f:
        tm = {}
        for k, v in test_metrics.items():
            if isinstance(v, torch.Tensor):
                tm[k] = float(v.item()) if v.numel() == 1 else v.tolist()
            else:
                tm[k] = float(v)
        json.dump(tm, f, indent=2)
    print(f"\nSaved to {MODEL_DIR}/test_metrics.json")
    print("Done.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()

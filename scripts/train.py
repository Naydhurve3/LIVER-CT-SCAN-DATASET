"""
Training CLI for Liver Tumor Segmentation.

Trains a model and saves checkpoints to MODELS_DIR/:
  - best_model.pth       (best validation dice)
  - checkpoint_epochN.pth (every ~10 epochs)
  - final_model.pth      (last epoch)
  - history.json         (training history)

Use --resume to continue from any saved checkpoint.

Usage:
    python scripts/train.py --model mobilenetv2_unet --epochs 50 --batch-size 8
    python scripts/train.py --model mobilenetv2_unet --resume models/best_model.pth
    python scripts/train.py --help
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEVICE, TRAIN_CONFIG_2D, PHASE4_RESEARCH_CONFIG, print_device_info, load_yaml_config
from src.utils import set_seed, logger, setup_logging
from src.data_loader import DatasetConfig, DataPathManager, VolumeWiseSplitter, create_2d_dataloaders
from src.preprocessing import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.models import create_model, count_params
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train liver tumor segmentation model")
    parser.add_argument("--model", type=str, default="mobilenetv2_unet",
                        choices=["mobilenetv2_unet", "all"],
                        help="Model architecture to train")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG_2D["num_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG_2D["batch_size"],
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG_2D["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision", default=True,
                        help="Disable mixed precision training")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained weights (default)")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Train from scratch (random init)")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Number of linear warmup epochs")
    parser.add_argument("--focal", dest="use_focal", action="store_true", default=True,
                        help="Use Focal Loss (recommended for class imbalance)")
    parser.add_argument("--no-focal", dest="use_focal", action="store_false",
                        help="Use BCE+Dice Combined Loss instead of Focal")
    parser.add_argument("--pos-weight", type=float, default=10.0,
                        help="Positive class weight for BCE loss (only used with --no-focal)")
    parser.add_argument("--no-aug", dest="use_aug", action="store_false", default=True,
                        help="Disable data augmentation")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for model checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    print_device_info()
    setup_logging()
    set_seed(args.seed)

    logger.info(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    path_manager = DataPathManager()
    logger.info("Building volume index...")
    volume_index = path_manager.build_index()

    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(DatasetConfig.SPLITS_DIR)
    logger.info(f"Loaded splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
    if args.use_aug:
        transform_train = AugmentedPreprocessingTransform(
            target_size=(256, 256), hu_low=-100, hu_high=400, clahe=clahe,
        )
    else:
        transform_train = PreprocessingTransform(
            target_size=(256, 256), hu_low=-100, hu_high=400,
        )
    transform_val = PreprocessingTransform(
        target_size=(256, 256), hu_low=-100, hu_high=400,
    )

    train_loader, val_loader, test_loader = create_2d_dataloaders(
        volume_index, splits['train'], splits['val'], splits['test'],
        batch_size=args.batch_size,
        transform_train=transform_train,
        transform_val=transform_val,
    )

    model = create_model(args.model, in_channels=1, out_channels=1, pretrained=args.pretrained)
    model = model.to(DEVICE)
    logger.info(f"Model parameters: {count_params(model):,}")

    output_dir = args.output_dir or "models"
    train_config = dict(PHASE4_RESEARCH_CONFIG)
    train_config['use_focal'] = args.use_focal
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        mixed_precision=args.mixed_precision,
        output_dir=output_dir,
    )

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume, model)
        logger.info("Checkpoint loaded. Continuing training...")

    logger.info("Starting training...")
    trainer.train(warmup_epochs=args.warmup_epochs)
    logger.info(f"Training complete. Best val dice: {trainer.best_val_dice:.4f}")

    import json
    history_path = Path(output_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Dice: {test_metrics['dice']:.4f}, IoU: {test_metrics['iou']:.4f}")


if __name__ == "__main__":
    main()

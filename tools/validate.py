import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from src.framework.core.config import build_experiment_config
from src.framework.core.reproducibility import set_seed
from src.framework.core.factory import build_model
from src.framework.data.lits_dataset import DataPathManager, VolumeWiseSplitter, create_2d_dataloaders
from src.framework.data.transforms import PreprocessingTransform, AugmentedPreprocessingTransform, CLAHEProcessor
from src.framework.models.mobilenetv2_unet import count_params, MobileNetV2UNet
from src.framework.training.trainer import Trainer
from src.framework.evaluation.metrics import dice_coefficient, iou_score
from src.framework.evaluation.surface_metrics import hausdorff_distance_95, average_surface_distance
from src.framework.utils.gpu_utils import DEVICE
from src.framework.utils.logging_utils import setup_logging

LOGGER = setup_logging(__name__)


def load_checkpoint_to_model(checkpoint_path, model_class, model_kwargs=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        model_kwargs = model_kwargs or {"in_channels": 1, "out_channels": 1}
        model = model_class(**model_kwargs)
        model.load_state_dict(ckpt)
        return model.to(DEVICE)
    return ckpt.to(DEVICE) if hasattr(ckpt, "to") else ckpt


def validate_checkpoint(checkpoint_path, val_loader, model_kwargs=None):
    model = load_checkpoint_to_model(checkpoint_path, MobileNetV2UNet, model_kwargs)
    model.eval()
    dices, ious = [], []
    start = time.time()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            targets = batch["mask"].to(DEVICE)
            preds = model(images)
            pred_binary = (preds > 0.5).float()
            dices.append(dice_coefficient(pred_binary, targets).item())
            ious.append(iou_score(pred_binary, targets).item())
    elapsed = time.time() - start
    n_slices = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 0
    return {
        "dice": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "iou": float(np.mean(ious)),
        "iou_std": float(np.std(ious)),
        "n_samples": len(dices),
        "inference_time": round(elapsed, 2),
        "inference_per_slice_ms": round(elapsed / n_slices * 1000, 2) if n_slices else 0,
    }


def run_short_training(epochs=1):
    set_seed(42)
    path_manager = DataPathManager(
        images_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset",
        masks_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks",
    )
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(Path("data/splits"))
    clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
    transform_train = AugmentedPreprocessingTransform(
        target_size=(256, 256), hu_low=-100, hu_high=400, clahe=clahe,
    )
    transform_val = PreprocessingTransform(
        target_size=(256, 256), hu_low=-100, hu_high=400,
    )
    train_loader, val_loader, _ = create_2d_dataloaders(
        volume_index, splits.get('train', []), splits.get('val', []), splits.get('test', []),
        batch_size=8, transform_train=transform_train, transform_val=transform_val,
    )
    model = build_model({"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1})
    model = model.to(DEVICE)
    LOGGER.info(f"Model parameters: {count_params(model):,}")
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        num_epochs=epochs, mixed_precision=True,
    )
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    LOGGER.info(f"Training {epochs} epoch(s) completed in {elapsed:.2f}s")
    return {"train_time": round(elapsed, 2), "epochs": epochs, "samples": len(train_loader.dataset)}


def run_full_validation():
    print("\n" + "=" * 60)
    print("SPRINT 4: BASELINE VALIDATION")
    print("=" * 60)

    print("\n--- Step 1: Load data ---")
    path_manager = DataPathManager(
        images_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset",
        masks_dir=r"D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks",
    )
    volume_index = path_manager.build_index()
    splitter = VolumeWiseSplitter()
    splits = splitter.load_splits(Path("data/splits"))
    LOGGER.info(f"Volumes: {len(volume_index.get('volumes', []))}")
    LOGGER.info(f"Train: {len(splits.get('train', []))}, Val: {len(splits.get('val', []))}, Test: {len(splits.get('test', []))}")

    transform_val = PreprocessingTransform(target_size=(256, 256), hu_low=-100, hu_high=400)
    _, val_loader, _ = create_2d_dataloaders(
        volume_index, splits.get('train', [])[:5], splits.get('val', [])[:5], [],
        batch_size=4, transform_val=transform_val,
    )
    LOGGER.info(f"Val samples: {len(val_loader.dataset)}")

    print("\n--- Step 2: Validate old checkpoints ---")
    checkpoint_dir = Path("models/upre")
    ensemble_results = []
    for i, ckpt_path in enumerate(sorted(checkpoint_dir.glob("member_*.pth"))):
        LOGGER.info(f"Evaluating {ckpt_path.name}...")
        result = validate_checkpoint(str(ckpt_path), val_loader)
        ensemble_results.append(result)
        LOGGER.info(f"  Dice: {result['dice']:.4f} +/- {result['dice_std']:.4f}, IoU: {result['iou']:.4f}")
        LOGGER.info(f"  Inference: {result['inference_per_slice_ms']:.2f} ms/slice")

    if ensemble_results:
        dices = [r["dice"] for r in ensemble_results]
        LOGGER.info(f"\nEnsemble Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
        print(f"  Expected (old project): ~0.85-0.87")

    print("\n--- Step 3: Benchmark ---")
    model = MobileNetV2UNet(in_channels=1, out_channels=1).to(DEVICE)
    params = count_params(model)
    LOGGER.info(f"Model parameters: {params:,}")
    dummy = torch.randn(1, 1, 256, 256).to(DEVICE)
    with torch.cuda.amp.autocast(enabled=True):
        with torch.no_grad():
            _ = model(dummy)
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1e6
        LOGGER.info(f"Peak VRAM: {vram:.2f} MB (batch_size=1)")
        torch.cuda.reset_peak_memory_stats()

    print("\n--- Step 4: Short training test (1 epoch) ---")
    train_result = run_short_training(epochs=1)
    LOGGER.info(f"Training: {train_result['train_time']}s for {train_result['epochs']} epoch(s)")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_full_validation()

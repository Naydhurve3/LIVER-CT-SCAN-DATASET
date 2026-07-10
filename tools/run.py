import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def smoke():
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    print("\n[1/5] Importing core module...")
    from src.framework.core.constants import PROJECT_ROOT, DEFAULT_SEED
    from src.framework.core.exceptions import MedSegXError
    from src.framework.core.interfaces import Configurable, Trainable, Predictable
    from src.framework.core.registry import MODELS, LOSSES, METRICS, DATASETS
    print(f"  Registries: {len(MODELS.list())} models, {len(LOSSES.list())} losses, {len(METRICS.list())} metrics")
    print("  OK")

    print("\n[2/5] Loading config...")
    from src.framework.experiment import configure_dataset_environment
    from src.framework.core.config import build_experiment_config
    configure_dataset_environment()
    cfg = build_experiment_config("configs/experiments/baseline.yaml")
    print(f"  Experiment: {cfg.get('experiment', {}).get('name', 'unknown')}")
    print("  OK")

    print("\n[3/5] Building model...")
    from src.framework.core.factory import build_model
    model = build_model({"name": "mobilenetv2_unet", "in_channels": 1, "out_channels": 1,
                         "pretrained": False})
    print(f"  Model: mobilenetv2_unet")
    print("  OK")

    print("\n[4/5] Forward pass (random input)...")
    import torch
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")
    print("  OK")

    print("\n[5/5] Loading loss functions...")
    from src.framework.losses.dice_loss import DiceLoss
    from src.framework.losses.combined_loss import CombinedLoss
    from src.framework.losses.uwacl_v1 import UncertaintyWeightedLoss
    dl = DiceLoss()
    cl = CombinedLoss()
    uwl = UncertaintyWeightedLoss()
    target = (torch.rand(2, 1, 256, 256) > 0.5).float()
    print(f"  Dice: {dl(y, target):.4f}, Combined: {cl(y, target):.4f}, UWACL: {uwl(y, target):.4f}")
    print("  OK")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(smoke())

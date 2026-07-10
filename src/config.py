"""
Central configuration for the Liver Tumor Segmentation project.
Detects GPU, defines paths, CT windowing constants, and training parameters.
Supports YAML config loading for advanced experiment management.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch


# =============================================================================
# 1. GPU / DEVICE SETUP
# =============================================================================
_device_printed = False

def setup_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def print_device_info():
    global _device_printed
    if _device_printed:
        return
    _device_printed = True
    if torch.cuda.is_available():
        print("=== Device: CUDA ===")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM: {vram_gb:.1f} GB")
    else:
        print("=== Device: CPU (CUDA not available) ===")


DEVICE = setup_device()

# =============================================================================
# 2. PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRAIN_SPLIT_FILE = SPLITS_DIR / "train_volumes.txt"
VAL_SPLIT_FILE = SPLITS_DIR / "val_volumes.txt"
TEST_SPLIT_FILE = SPLITS_DIR / "test_volumes.txt"

STATISTICS_FILE = METADATA_DIR / "statistics.json"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. CT WINDOWING CONSTANTS (Hounsfield Units)
# =============================================================================
WINDOWS = {
    "liver": {"level": 30, "width": 150},
    "abdomen": {"level": 50, "width": 400},
    "bone": {"level": 400, "width": 1800},
    "lung": {"level": -600, "width": 1500},
}

# =============================================================================
# 4. CLASS MAPPING
# =============================================================================
CLASS_MAPPING = {
    0: "Background",
    1: "Liver",
    2: "Tumor",
}
NUM_CLASSES = 3
CLASS_WEIGHTS = torch.tensor([0.1, 0.5, 0.4], device=DEVICE)

# =============================================================================
# 5. TRAINING CONSTANTS
# =============================================================================
TRAIN_CONFIG = {
    "batch_size": 4,
    "patch_size": (64, 64, 64),
    "stride": (32, 32, 32),
    "num_workers": 2,
    "pin_memory": True,
    "learning_rate": 1e-3,
    "num_epochs": 100,
    "gradient_accumulation_steps": 2,
    "mixed_precision": True,
}

# =============================================================================
# 6. INTENSITY NORMALIZATION (LiTS 2017 typical stats)
# =============================================================================
NORMALIZATION = {
    "mean": 0.0,
    "std": 1.0,
    "clip_min": -200,
    "clip_max": 250,
}

# =============================================================================
# 7. DATA AUGMENTATION SETTINGS
# =============================================================================
AUGMENTATION = {
    "rotation_range": 10,
    "flip_probability": 0.5,
    "elastic_alpha": 50,
    "elastic_sigma": 5,
    "intensity_shift_range": 0.1,
    "intensity_scale_range": 0.1,
}

# =============================================================================
# 8. REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# 9. 2D TRAINING CONFIGURATION (for PNG slice pipeline)
# =============================================================================
TRAIN_CONFIG_2D = {
    "batch_size": 8,
    "image_size": (256, 256),
    "num_workers": 4,
    "pin_memory": True,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "mixed_precision": True,
}

# =============================================================================
# 10. PNG DATA PATHS
# =============================================================================
IMAGES_DIR = PROJECT_ROOT.parent.parent / "Dataset" / "Liver Img Dataset"
MASKS_DIR = PROJECT_ROOT.parent.parent / "Dataset" / "LiTS_masks"

# =============================================================================
# 11. PHASE 4 RESEARCH CONFIG (UP³RE-Net)
# =============================================================================
PHASE4_RESEARCH_CONFIG = {
    "input_size": (256, 256),
    "batch_size": 8,
    "num_workers": 4,
    "pin_memory": True,
    "lr": 1e-3,
    "epochs_stage1": 25,
    "epochs_stage2": 25,
    "patience": 10,
    "pos_weight": 10.0,
    "dice_weight": 0.5,
    "bce_weight": 0.5,
    "mixed_precision": True,
    "num_bagging": 3,
    "uwacl_beta": 5.0,
    "uwacl_tau": 0.1,
    "uncertainty_schedule": "linear_decay",
    "uncertainty_dir": "tmp/uncertainty",
    "models_dir": "models/upre",
    "outputs_dir": "outputs/research",
}

# =============================================================================
# 12. YAML CONFIG LOADER
# =============================================================================
def load_yaml_config(path: Path) -> Optional[Dict[str, Any]]:
    """Load configuration from a YAML file.

    Merges YAML values with existing defaults where applicable.

    Args:
        path: Path to YAML config file.

    Returns:
        Config dict or None if file not found.
    """
    try:
        import yaml
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"[CONFIG] Loaded YAML config: {path}")
        return config
    except ImportError:
        print("[CONFIG] PyYAML not installed. Install with: pip install pyyaml")
        return None
    except FileNotFoundError:
        print(f"[CONFIG] Config file not found: {path}")
        return None
    except yaml.YAMLError as e:
        print(f"[CONFIG] Error parsing YAML: {e}")
        return None

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    print(f"  DEVICE: {DEVICE}")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  IMAGES_DIR: {IMAGES_DIR}  exists={IMAGES_DIR.exists()}")
    print(f"  MASKS_DIR: {MASKS_DIR}  exists={MASKS_DIR.exists()}")
    print(f"  TRAIN_CONFIG_2D: {TRAIN_CONFIG_2D}")
    print("  OK")

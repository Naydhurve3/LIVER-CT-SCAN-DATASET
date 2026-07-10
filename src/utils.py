"""
Utility functions: logging, error handling, progress bars, parallel loading, seeding.
"""
import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DEVICE, RANDOM_SEED


# =============================================================================
# 1. LOGGING
# =============================================================================
def setup_logging(name: str = "liver_project", level: int = logging.INFO) -> logging.Logger:
    """Configure a logger with timestamps and consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


logger = setup_logging()


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behaviour (may be slower)
        # torch.backends.cudnn.deterministic = True
    logger.info(f"Random seed set to {seed}")


# =============================================================================
# 3. FILE / PATH HELPERS
# =============================================================================
def check_file_exists(path: Path) -> bool:
    """Check if a file exists; log a warning if not."""
    exists = path.exists()
    if not exists:
        logger.warning(f"File not found: {path}")
    return exists


def load_volume_safe(path: Path) -> Optional[np.ndarray]:
    """Load a NIfTI volume with error handling. Returns None on failure."""
    import nibabel as nib
    try:
        if not check_file_exists(path):
            return None
        img = nib.load(str(path))
        data = img.get_fdata()
        logger.info(f"Loaded: {path.name} — shape={data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


def save_json(data, path: Path) -> None:
    """Save data as JSON with error handling."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON: {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}")


def load_json(path: Path):
    """Load JSON with error handling."""
    try:
        if not check_file_exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {path}: {e}")
        return {}


def load_split_ids(split_path: Path) -> List[str]:
    """Load volume IDs from a split file (one ID per line)."""
    try:
        if not check_file_exists(split_path):
            return []
        with open(split_path, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(ids)} IDs from {split_path.name}")
        return ids
    except Exception as e:
        logger.error(f"Failed to load split {split_path}: {e}")
        return []


# =============================================================================
# 4. PARALLEL LOADING
# =============================================================================
def load_volumes_parallel(
    volume_ids: List[str],
    data_dir: Path,
    suffix: str = ".nii",
    max_workers: int = 4,
    show_progress: bool = True,
) -> dict:
    """Load multiple volumes in parallel with a progress bar."""
    results = {}
    iterator = tqdm(volume_ids, desc="Loading volumes") if show_progress else volume_ids

    def _load_single(vid: str) -> Tuple[str, Optional[np.ndarray]]:
        vol_path = data_dir / f"{vid}{suffix}"
        if not vol_path.exists():
            vol_path = data_dir / f"{vid}.nii.gz"
        data = load_volume_safe(vol_path)
        return vid, data

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_single, vid): vid for vid in volume_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel load") if show_progress else futures:
            vid, data = future.result()
            if data is not None:
                results[vid] = data

    logger.info(f"Successfully loaded {len(results)}/{len(volume_ids)} volumes")
    return results


# =============================================================================
# 5. GPU TENSOR HELPERS
# =============================================================================
def to_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Move tensor to the configured device."""
    if device is None:
        device = DEVICE
    return tensor.to(device, non_blocking=True)


def numpy_to_gpu(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to GPU tensor in one step."""
    tensor = torch.from_numpy(arr).to(dtype=dtype)
    return to_device(tensor)


# =============================================================================
# 6. MEMORY MANAGEMENT
# =============================================================================
def clear_gpu_cache() -> None:
    """Clear GPU memory cache. Call between heavy operations."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")


def print_gpu_memory() -> None:
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU Mem — Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    setup_logging()
    logger.info("Logger OK")
    set_seed()
    print(f"  DEVICE: {DEVICE}")
    print("  OK")

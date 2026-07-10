"""
GPU utilities for tensor conversion and memory management.
Optimized for NVIDIA RTX 3050 Ti (4GB VRAM).
"""
import torch
import numpy as np
from typing import Optional


# ============================================================================
# DEVICE SETUP
# ============================================================================
def setup_device(verbose: bool = True) -> torch.device:
    """Detect and return the best available device.

    Returns CUDA device if available, CPU otherwise.
    Enables cuDNN benchmark mode for GPU performance.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("[GPU] Device: CUDA")
            print(f"[GPU] GPU: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] VRAM: {vram_gb:.2f} GB")
            print(f"[GPU] CUDA: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        if verbose:
            print("[GPU] Device: CPU (CUDA not available)")
    return device


DEVICE = setup_device(verbose=False)


# ============================================================================
# TENSOR CONVERSION
# ============================================================================
def to_tensor(arr, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy array to tensor on the active device.

    Args:
        arr: numpy array or list convertible to array.
        dtype: Target tensor dtype (default float32).

    Returns:
        Tensor on DEVICE.
    """
    t = torch.from_numpy(np.asarray(arr)).to(dtype=dtype)
    return t.to(DEVICE) if DEVICE.type == "cuda" else t


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array (moves from GPU to CPU if needed).

    Args:
        tensor: PyTorch tensor (could be on CPU or GPU).

    Returns:
        numpy array.
    """
    return tensor.cpu().numpy() if tensor.device.type == "cuda" else tensor.numpy()


def batch_to_tensor(batch_list, dtype=torch.float32) -> torch.Tensor:
    """Convert list of numpy arrays to batched tensor on device.

    Args:
        batch_list: List of numpy arrays.
        dtype: Target dtype.

    Returns:
        Stacked tensor on DEVICE: (B, ...) shape.
    """
    stacked = np.stack(batch_list).astype(np.float32)
    t = torch.from_numpy(stacked).to(dtype=dtype)
    return t.to(DEVICE) if DEVICE.type == "cuda" else t


# ============================================================================
# GPU MEMORY MANAGEMENT
# ============================================================================
def gpu_report() -> str:
    """Get current GPU memory usage as a formatted string.

    Returns:
        String like "GPU Memory: 0.50 GB allocated, 1.20 GB reserved, 4.00 GB total"
    """
    if DEVICE.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return (f"GPU Memory: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, {total:.2f}GB total")
    return "GPU: Not available (using CPU)"


def gpu_clear():
    """Clear GPU memory cache. Call between heavy operations.

    Frees unused cached memory but does not empty allocated tensors
    that are still referenced.
    """
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory():
    """Print current GPU memory usage using gpu_report()."""
    print(f"   {gpu_report()}")

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    print(f"  DEVICE: {setup_device(verbose=False)}")
    print(f"  gpu_report(): {gpu_report()}")
    import numpy as np
    t = to_tensor(np.ones((2, 3)))
    print(f"  to_tensor: shape={t.shape}, dtype={t.dtype}")
    print("  OK")

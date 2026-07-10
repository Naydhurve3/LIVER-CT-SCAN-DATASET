import torch
import numpy as np


def setup_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] VRAM: {vram_gb:.2f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        if verbose:
            print("[GPU] Device: CPU (CUDA not available)")
    return device


def to_tensor(arr, dtype=torch.float32, device=None) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr)).to(dtype=dtype)
    if device:
        t = t.to(device)
    return t


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy() if tensor.device.type == "cuda" else tensor.numpy()


def batch_to_tensor(batch_list, dtype=torch.float32) -> torch.Tensor:
    stacked = np.stack(batch_list).astype(np.float32)
    return torch.from_numpy(stacked).to(dtype=dtype)


def gpu_report() -> str:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return (f"GPU: {allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, {total:.2f}GB total")
    return "GPU: Not available"


def gpu_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_memory():
    print(f"   {gpu_report()}")


DEVICE = setup_device(verbose=False)

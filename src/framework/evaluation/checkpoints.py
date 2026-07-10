from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


STATE_KEYS = ("model_state", "model_state_dict", "state_dict")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(payload: Any) -> Tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a state dictionary")
    for key in STATE_KEYS:
        if key in payload:
            return payload[key], {k: v for k, v in payload.items() if k != key}
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload, {}
    raise ValueError(f"No model state found; expected one of {STATE_KEYS}")


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str | Path,
                               device: torch.device | str = "cpu",
                               strict: bool = True) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device, weights_only=True)
    state_dict, metadata = extract_state_dict(payload)
    cleaned = {(key[7:] if key.startswith("module.") else key): value
               for key, value in state_dict.items()}
    incompatibility = model.load_state_dict(cleaned, strict=strict)
    model.to(device)
    return {
        "checkpoint": str(path.resolve()), "sha256": sha256_file(path),
        "metadata": metadata, "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
    }


def split_hashes(split_dir: str | Path) -> Dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(Path(split_dir).glob("*.txt"))}


def environment_snapshot() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "git_commit": commit,
    }


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

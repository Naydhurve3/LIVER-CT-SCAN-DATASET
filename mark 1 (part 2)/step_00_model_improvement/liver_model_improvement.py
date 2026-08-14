"""Shared library for the Mark Part 2 Step 00 model-improvement programme.

All candidate arms (Control, C1 highres-ROI, C2 analog sampler, C3 capped recall
loss) reuse the real frozen interfaces described in the project contract:

- Loader:  ``VerifiedManifestDataset``
- Model:   ``MobileNetV2UNet``
- Loss:    ``FocalDiceLoss`` (control / C1 / C2) and ``StabilityBoundedRecallLoss`` (C3)
- ROI:     frozen predicted-liver rule ``threshold=0.50, padding=16, largest_3d``

Everything is parameterised from ``Configuration`` so phases stay reproducible.
This module is intentionally free of data-path hard-coding except for the
corrected LiTS build root and the authoritative manifest hash, which are frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


DATASET_ROOT = Path(
    r"D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver\02_staging"
    r"\build_corrected_20260713_214847_v2"
)
EXPECTED_MANIFEST_SHA256 = (
    "575a6fc391d63dc9bbbbb3317efe4d0f65edd57457ae63c1bfc6c2c615861889"
)
SEED = 42


# --------------------------------------------------------------------------- #
# Hash, json and file helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Union[str, Path]):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Image / mask resizing helpers (frozen interpolation rules)
# --------------------------------------------------------------------------- #
def resize_float(array: np.ndarray, size: Tuple[int, int]):
    return np.asarray(
        Image.fromarray(array.astype(np.float32), mode="F").resize(
            size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def resize_mask(array: np.ndarray, size: Tuple[int, int]):
    return np.asarray(
        Image.fromarray(array.astype(np.uint8) * 255).resize(
            size, Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 0


# --------------------------------------------------------------------------- #
# Deterministic seeding
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = SEED) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
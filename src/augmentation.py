"""
Data augmentation transforms for 3D medical image volumes.
Supports flips, rotations, elastic deformation, and intensity perturbations.
"""
import random
from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import AUGMENTATION
from src.utils import logger


class RandomFlip3D:
    """Randomly flip the 3D volume along specified axes."""

    def __init__(self, axes: Tuple[int, ...] = (1, 2), p: float = 0.5):
        """
        Args:
            axes: Axes along which to flip (1=height, 2=width, 0=depth).
            p: Probability of flipping.
        """
        self.axes = axes
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        for axis in self.axes:
            if random.random() < self.p:
                # Image has shape (C, D, H, W), mask has shape (D, H, W)
                # axis 0=depth, 1=height, 2=width in mask coords
                # image dims are offset by 1 for channel
                image = torch.flip(image, dims=[axis + 1])
                mask = torch.flip(mask, dims=[axis])
        return {"image": image, "mask": mask}


class RandomRotate90:
    """Random 90-degree rotation in axial plane."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            # Rotate spatial dims (H, W) = dims 2, 3 for image, dims 1, 2 for mask
            image = torch.rot90(image, k, dims=[2, 3])
            mask = torch.rot90(mask, k, dims=[1, 2])
        return {"image": image, "mask": mask}


class RandomIntensityShift:
    """Randomly shift and scale intensity values."""

    def __init__(
        self,
        shift_range: float = AUGMENTATION["intensity_shift_range"],
        scale_range: float = AUGMENTATION["intensity_scale_range"],
        p: float = 0.5,
    ):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            shift = random.uniform(-self.shift_range, self.shift_range)
            scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            image = image * scale + shift
        return {"image": image, "mask": mask}


class RandomNoise:
    """Add Gaussian noise to the volume."""

    def __init__(self, noise_std: float = 0.01, p: float = 0.3):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        return {"image": image, "mask": mask}


class RandomElasticDeformation3D:
    """
    Elastic deformation of 3D volumes using displacement fields.
    Simulates realistic tissue deformation.
    """

    def __init__(
        self,
        alpha: float = AUGMENTATION["elastic_alpha"],
        sigma: float = AUGMENTATION["elastic_sigma"],
        p: float = 0.3,
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if random.random() >= self.p:
            return {"image": image, "mask": mask}

        img_np = image.squeeze(0).cpu().numpy()  # (D, H, W)
        mask_np = mask.cpu().numpy()  # (D, H, W)

        shape = img_np.shape
        if len(shape) != 3:
            return {"image": image, "mask": mask}

        # Generate random displacement fields
        dx = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dy = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dz = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha

        # Create meshgrid
        z, y, x = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]),
            np.arange(shape[2]), indexing="ij"
        )

        # Apply displacement
        indices = np.array(
            [np.clip(z + dz, 0, shape[0] - 1),
             np.clip(y + dy, 0, shape[1] - 1),
             np.clip(x + dx, 0, shape[2] - 1)]
        )

        # Warp image and mask
        img_warped = map_coordinates(img_np, indices, order=1, mode="nearest")
        mask_warped = map_coordinates(mask_np.astype(float), indices, order=0, mode="nearest")

        return {
            "image": torch.from_numpy(img_warped).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_warped.round()).long(),
        }


class Compose3D:
    """Compose multiple 3D augmentation transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        for t in self.transforms:
            result = t(image, mask)
            image = result["image"]
            mask = result["mask"]
        return {"image": image, "mask": mask}


# =============================================================================
# FACTORY: Common augmentation pipelines
# =============================================================================
def get_train_transforms() -> Compose3D:
    """Standard training augmentation pipeline."""
    return Compose3D([
        RandomFlip3D(axes=(1, 2), p=0.5),
        RandomRotate90(p=0.5),
        RandomIntensityShift(shift_range=0.1, scale_range=0.1, p=0.5),
        RandomNoise(noise_std=0.01, p=0.3),
    ])


def get_val_transforms():
    """No augmentation for validation."""
    return None


def get_test_transforms():
    """No augmentation for testing."""
    return None

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    x = torch.randn(1, 8, 64, 64)
    m = torch.randint(0, 2, (8, 64, 64))
    t = get_train_transforms()
    r = t(x, m)
    print(f"  Augmented: image={r['image'].shape}, mask={r['mask'].shape}")
    print(f"  Transforms: Flip3D, Rotate90, IntensityShift, Noise, Elastic, Compose3D")
    print("  OK")

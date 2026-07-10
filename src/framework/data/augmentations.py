import random
from typing import Tuple

import torch
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


class RandomFlip3D:
    def __init__(self, axes: Tuple[int, ...] = (1, 2), p: float = 0.5):
        self.axes = axes
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        for axis in self.axes:
            if random.random() < self.p:
                image = torch.flip(image, dims=[axis + 1])
                mask = torch.flip(mask, dims=[axis])
        return {"image": image, "mask": mask}


class RandomRotate90:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[2, 3])
            mask = torch.rot90(mask, k, dims=[1, 2])
        return {"image": image, "mask": mask}


class RandomIntensityShift:
    def __init__(self, shift_range: float = 0.1, scale_range: float = 0.1, p: float = 0.5):
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p:
            shift = random.uniform(-self.shift_range, self.shift_range)
            scale = random.uniform(1 - self.scale_range, 1 + self.scale_range)
            image = image * scale + shift
        return {"image": image, "mask": mask}


class RandomNoise:
    def __init__(self, noise_std: float = 0.01, p: float = 0.3):
        self.noise_std = noise_std
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        return {"image": image, "mask": mask}


class RandomElasticDeformation3D:
    def __init__(self, alpha: float = 50, sigma: float = 5, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() >= self.p:
            return {"image": image, "mask": mask}
        img_np = image.squeeze(0).cpu().numpy()
        mask_np = mask.cpu().numpy()
        shape = img_np.shape
        if len(shape) != 3:
            return {"image": image, "mask": mask}
        dx = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dy = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        dz = gaussian_filter(np.random.randn(*shape), self.sigma) * self.alpha
        z, y, x = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]),
            np.arange(shape[2]), indexing="ij"
        )
        indices = np.array([
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1),
        ])
        img_warped = map_coordinates(img_np, indices, order=1, mode="nearest")
        mask_warped = map_coordinates(mask_np.astype(float), indices, order=0, mode="nearest")
        return {
            "image": torch.from_numpy(img_warped).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask_warped.round()).long(),
        }


class Compose3D:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        for t in self.transforms:
            result = t(image, mask)
            image = result["image"]
            mask = result["mask"]
        return {"image": image, "mask": mask}


def get_train_transforms():
    return Compose3D([
        RandomFlip3D(axes=(1, 2), p=0.5),
        RandomRotate90(p=0.5),
        RandomIntensityShift(shift_range=0.1, scale_range=0.1, p=0.5),
        RandomNoise(noise_std=0.01, p=0.3),
    ])


def get_val_transforms():
    return None


def get_test_transforms():
    return None

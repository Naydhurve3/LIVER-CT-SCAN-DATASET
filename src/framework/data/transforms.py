from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def hu_window_cpu(img: np.ndarray, low: int = -100, high: int = 400) -> np.ndarray:
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    scale_low = low / 500.0
    scale_high = high / 500.0
    clipped = np.clip(img_u8, scale_low * 255, scale_high * 255)
    normalized = (clipped - scale_low * 255) / (scale_high * 255 - scale_low * 255)
    return normalized.astype(np.float32)


def hu_window_batch(batch: torch.Tensor, low: float = -100, high: float = 400) -> torch.Tensor:
    scale_low = low / 500.0
    scale_high = high / 500.0
    clipped = torch.clamp(batch, scale_low, scale_high)
    normalized = (clipped - scale_low) / (scale_high - scale_low)
    return normalized


def resize_image(img: np.ndarray, size: Tuple[int, int], method: int = cv2.INTER_LINEAR) -> np.ndarray:
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    resized = cv2.resize(img_u8, size, interpolation=method)
    return resized.astype(np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    mask_u8 = (mask * 255).astype(np.uint8)
    resized = cv2.resize(mask_u8, size, interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.float32)


def apply_hu_window(volume: np.ndarray, window_name: str = "liver",
                     level: Optional[float] = None, width: Optional[float] = None) -> np.ndarray:
    windows = {
        "liver": {"level": 30, "width": 150},
        "abdomen": {"level": 50, "width": 400},
        "bone": {"level": 400, "width": 1800},
        "lung": {"level": -600, "width": 1500},
    }
    if level is None or width is None:
        if window_name not in windows:
            window_name = "liver"
        level = windows[window_name]["level"]
        width = windows[window_name]["width"]
    lower = level - width / 2
    upper = level + width / 2
    windowed = np.clip(volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)


def normalize_volume(volume: np.ndarray, clip_range: Tuple[float, float] = (-200, 250),
                     method: str = "zscore") -> np.ndarray:
    volume = np.clip(volume, clip_range[0], clip_range[1]).astype(np.float32)
    if method == "zscore":
        mean = volume.mean()
        std = volume.std()
        if std > 1e-8:
            volume = (volume - mean) / std
        else:
            volume = volume - mean
    elif method == "minmax":
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 1e-8:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = volume - vmin
    return volume.astype(np.float32)


class CLAHEProcessor:
    def __init__(self, clip: float = 2.0, grid: Tuple[int, int] = (8, 8)):
        self.clip = clip
        self.grid = grid
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)

    def __getstate__(self):
        return {'clip': self.clip, 'grid': self.grid}

    def __setstate__(self, state):
        self.__init__(state['clip'], state['grid'])

    def apply(self, img: np.ndarray) -> np.ndarray:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        enhanced = self._clahe.apply(img_u8)
        return enhanced.astype(np.float32) / 255.0

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(0, batch.size(0), 8):
            chunk = batch[i:i + 8]
            chunk_np = chunk.cpu().numpy()
            chunk_enhanced = np.stack([self.apply(c) for c in chunk_np])
            results.append(torch.from_numpy(chunk_enhanced).to(batch.device))
        return torch.cat(results, dim=0)


class PreprocessingTransform:
    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 hu_low: int = -100, hu_high: int = 400,
                 clahe: Optional[CLAHEProcessor] = None,
                 apply_hu_window: bool = False):
        self.target_size = target_size
        self.hu_low = hu_low
        self.hu_high = hu_high
        self.clahe = clahe
        self.apply_hu_window = apply_hu_window

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        if self.apply_hu_window:
            img = hu_window_cpu(img, self.hu_low, self.hu_high)
        img = resize_image(img, self.target_size)
        mask = resize_mask(mask, self.target_size)
        if self.clahe is not None:
            img = self.clahe.apply(img)
        return img, mask


class AugmentedPreprocessingTransform:
    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 hu_low: int = -100, hu_high: int = 400,
                 clahe: Optional[CLAHEProcessor] = None,
                 flip_prob: float = 0.5, shift_range: float = 0.1,
                 apply_hu_window: bool = False):
        self.target_size = target_size
        self.hu_low = hu_low
        self.hu_high = hu_high
        self.clahe = clahe
        self.flip_prob = flip_prob
        self.shift_range = shift_range
        self.apply_hu_window = apply_hu_window

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        import random
        if self.apply_hu_window:
            img = hu_window_cpu(img, self.hu_low, self.hu_high)
        img = resize_image(img, self.target_size)
        mask = resize_mask(mask, self.target_size)
        if self.clahe is not None:
            img = self.clahe.apply(img)
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < 0.5:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            img = np.clip(img + shift, 0.0, 1.0)
        return img, mask

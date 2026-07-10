"""
Preprocessing pipeline: HU windowing, normalization, patch extraction, class imbalance handling.
"""
import random
from typing import Tuple, Optional, List
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DEVICE, NORMALIZATION, WINDOWS, TRAIN_CONFIG, NUM_CLASSES
from src.utils import logger


# =============================================================================
# 1. CT WINDOWING (Hounsfield Unit clamping)
# =============================================================================
def apply_hu_window(
    volume: np.ndarray,
    window_name: str = "liver",
    level: Optional[float] = None,
    width: Optional[float] = None,
) -> np.ndarray:
    """
    Apply a standard CT window to the volume.

    Args:
        volume: 3D CT volume in Hounsfield Units.
        window_name: Preset name ('liver', 'abdomen', 'bone', 'lung').
        level: Window level (center) — overrides preset.
        width: Window width — overrides preset.

    Returns:
        Windowed volume with values in [0, 1].
    """
    if level is None or width is None:
        if window_name not in WINDOWS:
            logger.warning(f"Unknown window '{window_name}', using 'liver'")
            window_name = "liver"
        level = WINDOWS[window_name]["level"]
        width = WINDOWS[window_name]["width"]

    lower = level - width / 2
    upper = level + width / 2
    windowed = np.clip(volume, lower, upper)
    # Normalize to [0, 1]
    windowed = (windowed - lower) / (upper - lower)
    return windowed.astype(np.float32)


# =============================================================================
# 2. INTENSITY NORMALIZATION
# =============================================================================
def normalize_volume(
    volume: np.ndarray,
    clip_range: Tuple[float, float] = (NORMALIZATION["clip_min"], NORMALIZATION["clip_max"]),
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize CT volume intensity.

    Args:
        volume: 3D CT volume.
        clip_range: HU range for clipping.
        method: 'zscore' or 'minmax'.

    Returns:
        Normalized volume.
    """
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
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return volume.astype(np.float32)


# =============================================================================
# 3. RESAMPLE TO SPACING
# =============================================================================
def resample_to_spacing(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> np.ndarray:
    """Resample volume to a target voxel spacing using zoom."""
    scale_factors = [
        o / t for o, t in zip(original_spacing, target_spacing)
    ]
    new_shape = [
        int(round(s * z)) for s, z in zip(volume.shape, scale_factors)
    ]
    factors = [n / o for n, o in zip(new_shape, volume.shape)]
    resampled = zoom(volume, factors, order=order)
    return resampled.astype(np.float32)


# =============================================================================
# 4. 3D PATCH EXTRACTION
# =============================================================================
def extract_patches_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int] = TRAIN_CONFIG["patch_size"],
    stride: Tuple[int, int, int] = TRAIN_CONFIG["stride"],
    min_tumor_fraction: float = 0.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract 3D patches from volume and mask with optional tumor-focused sampling.

    Args:
        volume: 3D array (D, H, W).
        mask: 3D label array (D, H, W).
        patch_size: (D, H, W) patch dimensions.
        stride: (D, H, W) stride for sliding window.
        min_tumor_fraction: Minimum fraction of tumor pixels in a patch to keep it.
                            Set > 0 to focus on tumor regions.

    Returns:
        (volume_patches, mask_patches) lists.
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    vol_patches = []
    mask_patches = []

    for d in range(0, D - pd + 1, sd):
        for h in range(0, H - ph + 1, sh):
            for w in range(0, W - pw + 1, sw):
                vol_patch = volume[d:d+pd, h:h+ph, w:w+pw]
                mask_patch = mask[d:d+pd, h:h+ph, w:w+pw]

                # Skip patches with insufficient tumor content
                if min_tumor_fraction > 0:
                    tumor_pixels = (mask_patch == 2).sum()
                    total_pixels = mask_patch.size
                    if tumor_pixels / total_pixels < min_tumor_fraction:
                        # Still include random patches from non-tumor regions
                        # to maintain class balance — controlled by sampling rate
                        if np.random.random() > 0.2:  # Keep 20% of non-tumor patches
                            continue

                vol_patches.append(vol_patch.copy())
                mask_patches.append(mask_patch.copy())

    logger.debug(f"Extracted {len(vol_patches)} patches (tumor fraction >= {min_tumor_fraction})")
    return vol_patches, mask_patches


def extract_patches_balanced(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int] = TRAIN_CONFIG["patch_size"],
    num_patches: int = 100,
    tumor_ratio: float = 0.5,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract a fixed number of patches with balanced tumor/non-tumor ratio.

    Args:
        volume: 3D array (D, H, W).
        mask: 3D label array (D, H, W).
        patch_size: (D, H, W) patch dimensions.
        num_patches: Total number of patches to extract.
        tumor_ratio: Fraction of patches that should contain tumor.

    Returns:
        (volume_patches, mask_patches) lists.
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    num_tumor = int(num_patches * tumor_ratio)
    num_bg = num_patches - num_tumor

    tumor_patches_v, tumor_patches_m = [], []
    bg_patches_v, bg_patches_m = [], []

    tumor_voxels = np.argwhere(mask == 2)

    # Extract tumor-centered patches
    if len(tumor_voxels) > 0:
        for _ in range(num_tumor * 3):  # Oversample to fill quota
            if len(tumor_patches_v) >= num_tumor:
                break
            center = tumor_voxels[np.random.randint(len(tumor_voxels))]
            d_start = max(0, center[0] - pd // 2)
            h_start = max(0, center[1] - ph // 2)
            w_start = max(0, center[2] - pw // 2)
            d_end = min(D, d_start + pd)
            h_end = min(H, h_start + ph)
            w_end = min(W, w_start + pw)

            if d_end - d_start < pd or h_end - h_start < ph or w_end - w_start < pw:
                continue

            vol_patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]
            mask_patch = mask[d_start:d_end, h_start:h_end, w_start:w_end]
            tumor_patches_v.append(vol_patch.copy())
            tumor_patches_m.append(mask_patch.copy())

    # Extract background patches (random locations)
    for _ in range(num_bg * 2):
        if len(bg_patches_v) >= num_bg:
            break
        d_start = np.random.randint(0, max(1, D - pd))
        h_start = np.random.randint(0, max(1, H - ph))
        w_start = np.random.randint(0, max(1, W - pw))
        vol_patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        mask_patch = mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        # Only keep if no tumor (or very little)
        if (mask_patch == 2).sum() < 10:
            bg_patches_v.append(vol_patch.copy())
            bg_patches_m.append(mask_patch.copy())

    # Combine and shuffle
    vol_patches = tumor_patches_v[:num_tumor] + bg_patches_v[:num_bg]
    mask_patches = tumor_patches_m[:num_tumor] + bg_patches_m[:num_bg]

    combined = list(zip(vol_patches, mask_patches))
    np.random.shuffle(combined)
    vol_patches, mask_patches = zip(*combined) if combined else ([], [])

    logger.info(f"Extracted {len(vol_patches)} balanced patches ({len(tumor_patches_v)} tumor, {len(bg_patches_v)} background)")
    return list(vol_patches), list(mask_patches)


# =============================================================================
# 5. CLASS IMBALANCE — Weight Maps
# =============================================================================
def compute_class_weights(mask: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for the Dice loss.
    Inverse frequency weighting.
    """
    class_counts = np.array([(mask == c).sum() for c in range(NUM_CLASSES)])
    class_counts = np.maximum(class_counts, 1)
    total = class_counts.sum()
    weights = total / (NUM_CLASSES * class_counts)
    weights = weights / weights.sum() * NUM_CLASSES  # Normalize so sum = NUM_CLASSES
    return torch.from_numpy(weights).float()


def one_hot_encode(mask: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Convert (B, D, H, W) label mask to (B, C, D, H, W) one-hot."""
    return F.one_hot(mask.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


# ============================================================================
# 6. 2D PREPROCESSING FUNCTIONS (for PNG slice pipeline)
# ============================================================================

def hu_window_cpu(img: np.ndarray, low: int = -100, high: int = 400) -> np.ndarray:
    """CPU HU windowing: clip to [low, high] HU, normalize to [0, 1].

    Standard liver window: [-100, 400] HU
    Captures liver tissue while filtering bone/air.

    Args:
        img: Input image normalized to [0, 1] range.
        low: HU lower bound (default -100).
        high: HU upper bound (default 400).

    Returns:
        Windowed image in [0, 1] as float32.
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    scale_low = low / 500.0
    scale_high = high / 500.0
    clipped = np.clip(img_u8, scale_low * 255, scale_high * 255)
    normalized = (clipped - scale_low * 255) / (scale_high * 255 - scale_low * 255)
    return normalized.astype(np.float32)


def hu_window_batch(batch: torch.Tensor, low: float = -100, high: float = 400) -> torch.Tensor:
    """GPU batch HU windowing.

    Args:
        batch: (B, H, W) tensor in range [0, 1].
        low: HU lower bound.
        high: HU upper bound.

    Returns:
        (B, H, W) tensor normalized to [0, 1].
    """
    scale_low = low / 500.0
    scale_high = high / 500.0
    clipped = torch.clamp(batch, scale_low, scale_high)
    normalized = (clipped - scale_low) / (scale_high - scale_low)
    return normalized


def resize_image(img: np.ndarray, size: Tuple[int, int],
                 method: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize image using OpenCV with bilinear interpolation.

    Args:
        img: Image in [0, 1] as float32.
        size: Target (width, height).
        method: OpenCV interpolation method.

    Returns:
        Resized image in [0, 1].
    """
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    resized = cv2.resize(img_u8, size, interpolation=method)
    return resized.astype(np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize binary mask using nearest neighbor (preserves values).

    Args:
        mask: Binary mask (0 or 1 values).
        size: Target (width, height).

    Returns:
        Resized binary mask (0 or 1).
    """
    mask_u8 = (mask * 255).astype(np.uint8)
    resized = cv2.resize(mask_u8, size, interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.float32)


class CLAHEProcessor:
    """Contrast Limited Adaptive Histogram Equalization for CT images.

    Enhances local contrast while limiting noise amplification.
    """

    def __init__(self, clip: float = 2.0, grid: Tuple[int, int] = (8, 8)):
        """Initialize CLAHE with clip limit and tile grid size.

        Args:
            clip: Clip limit (higher = more contrast).
            grid: Tile grid size (rows, cols).
        """
        self.clip = clip
        self.grid = grid
        self._clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)

    def __getstate__(self):
        return {'clip': self.clip, 'grid': self.grid}

    def __setstate__(self, state):
        self.__init__(state['clip'], state['grid'])

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a single 0-1 normalized image.

        Args:
            img: Image in [0, 1] range.

        Returns:
            CLAHE-enhanced image in [0, 1].
        """
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        enhanced = self._clahe.apply(img_u8)
        return enhanced.astype(np.float32) / 255.0

    def apply_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply CLAHE to a batch of images (processes in chunks on CPU).

        Args:
            batch: (B, H, W) tensor.

        Returns:
            (B, H, W) enhanced tensor.
        """
        from src.gpu_utils import to_numpy, to_tensor
        results = []
        for i in range(0, batch.size(0), 8):
            chunk = batch[i:i + 8]
            chunk_np = to_numpy(chunk)
            chunk_enhanced = np.stack([self.apply(c) for c in chunk_np])
            results.append(to_tensor(chunk_enhanced))
        return torch.cat(results, dim=0)


class PreprocessingTransform:
    """Callable transform for LiverTumor2DDataset: HU window → resize → optional CLAHE.

    Applies preprocessing to (img, mask) numpy pairs and returns
    preprocessed (img, mask) ready for tensor conversion.

    Args:
        target_size: (H, W) for output images/masks.
        hu_low: HU lower bound for windowing.
        hu_high: HU upper bound for windowing.
        clahe: Optional CLAHEProcessor instance. If provided, applies CLAHE after resize.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        hu_low: int = -100,
        hu_high: int = 400,
        clahe: Optional["CLAHEProcessor"] = None,
    ):
        self.target_size = target_size
        self.hu_low = hu_low
        self.hu_high = hu_high
        self.clahe = clahe

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = hu_window_cpu(img, self.hu_low, self.hu_high)
        img = resize_image(img, self.target_size)
        mask = resize_mask(mask, self.target_size)
        if self.clahe is not None:
            img = self.clahe.apply(img)
        return img, mask


class AugmentedPreprocessingTransform:
    """Training transform: preprocessing + augmentation for 2D slices.

    Applies HU windowing, resize, CLAHE, then random flips and intensity shifts.

    Args:
        target_size: (H, W) for output.
        hu_low, hu_high: HU window bounds.
        clahe: Optional CLAHEProcessor.
        flip_prob: Probability of horizontal flip.
        shift_range: Max intensity shift fraction.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        hu_low: int = -100,
        hu_high: int = 400,
        clahe: Optional["CLAHEProcessor"] = None,
        flip_prob: float = 0.5,
        shift_range: float = 0.1,
    ):
        self.target_size = target_size
        self.hu_low = hu_low
        self.hu_high = hu_high
        self.clahe = clahe
        self.flip_prob = flip_prob
        self.shift_range = shift_range

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img = hu_window_cpu(img, self.hu_low, self.hu_high)
        img = resize_image(img, self.target_size)
        mask = resize_mask(mask, self.target_size)
        if self.clahe is not None:
            img = self.clahe.apply(img)
        # Random horizontal flip
        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        # Random intensity shift
        if random.random() < 0.5:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            img = np.clip(img + shift, 0.0, 1.0)
        return img, mask


def preprocess_batch_gpu(
    images: torch.Tensor,
    masks: torch.Tensor,
    target_size: Tuple[int, int],
    hu_low: int = -100,
    hu_high: int = 400,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full GPU preprocessing pipeline: HU windowing + resize for a batch.

    Args:
        images: (B, H, W) tensor in [0, 1] range.
        masks: (B, H, W) tensor with binary values (0 or 1).
        target_size: (width, height) target resolution.
        hu_low: HU lower bound.
        hu_high: HU upper bound.

    Returns:
        (images_out, masks_out) tensors on the same device.
    """
    batch = hu_window_batch(images, hu_low, hu_high)
    images_out = F.interpolate(
        batch.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False
    ).squeeze(1)
    masks_out = F.interpolate(
        masks.unsqueeze(1), size=target_size, mode="nearest"
    ).squeeze(1)
    return images_out, masks_out

if __name__ == "__main__":
    print(f"=== {__file__} ===")
    import numpy as np
    vol = np.random.randn(16, 64, 64).astype(np.float32)
    win = apply_hu_window(vol, "liver")
    print(f"  HU window: {win.shape}, range=[{win.min():.3f}, {win.max():.3f}]")
    norm = normalize_volume(vol, method="zscore")
    print(f"  Normalize: {norm.shape}, mean={norm.mean():.4f}, std={norm.std():.4f}")
    patches_v, patches_m = extract_patches_3d(vol, (vol > 0).astype(int), patch_size=(8, 32, 32), stride=(4, 16, 16))
    print(f"  Patches: {len(patches_v)}")
    t = PreprocessingTransform()
    img, m = t(np.ones((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.float32))
    print(f"  Transform: img={img.shape}, mask={m.shape}")
    print("  OK")

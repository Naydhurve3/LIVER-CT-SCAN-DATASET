import numpy as np
from src.framework.data.transforms import hu_window_cpu, resize_image, resize_mask, PreprocessingTransform


def test_hu_window_output_range():
    img = np.random.rand(256, 256).astype(np.float32)
    result = hu_window_cpu(img, -100, 400)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_hu_window_extreme_values():
    img = np.ones((256, 256), dtype=np.float32)
    img[:100] = 0.0
    result = hu_window_cpu(img, -100, 400)
    assert result.shape == (256, 256)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_resize_image_shape():
    img = np.random.rand(512, 512).astype(np.float32)
    result = resize_image(img, (256, 256))
    assert result.shape == (256, 256)
    assert result.dtype == np.float32


def test_resize_mask_binary():
    mask = (np.random.rand(512, 512) > 0.5).astype(np.float32)
    result = resize_mask(mask, (256, 256))
    assert result.shape == (256, 256)
    assert set(np.unique(result)).issubset({0.0, 1.0})


def test_preprocessing_transform_pipeline():
    img = np.random.rand(512, 512).astype(np.float32)
    mask = (np.random.rand(512, 512) > 0.5).astype(np.float32)
    transform = PreprocessingTransform(target_size=(256, 256), hu_low=-100, hu_high=400)
    img_out, mask_out = transform(img, mask)
    assert img_out.shape == (256, 256)
    assert mask_out.shape == (256, 256)
    assert img_out.dtype == np.float32
    assert mask_out.dtype == np.float32


def test_png_preprocessing_does_not_apply_hu_window_by_default():
    img = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    mask = np.zeros((8, 8), dtype=np.float32)
    transform = PreprocessingTransform(target_size=(8, 8))
    img_out, _ = transform(img, mask)
    assert np.allclose(img_out, img, atol=1 / 255)

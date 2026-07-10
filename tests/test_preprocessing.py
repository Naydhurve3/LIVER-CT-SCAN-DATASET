import unittest
import numpy as np
import torch
from src.preprocessing import (
    apply_hu_window, normalize_volume, hu_window_cpu, hu_window_batch,
    resize_image, resize_mask, CLAHEProcessor, PreprocessingTransform,
    AugmentedPreprocessingTransform, extract_patches_3d, extract_patches_balanced,
    compute_class_weights, one_hot_encode, preprocess_batch_gpu,
)


class TestHUWindowing(unittest.TestCase):
    def test_apply_hu_window_liver_default(self):
        vol = np.random.uniform(-1000, 1000, (10, 64, 64))
        result = apply_hu_window(vol, window_name="liver")
        self.assertEqual(result.shape, vol.shape)
        self.assertAlmostEqual(result.min(), 0.0, places=5)
        self.assertAlmostEqual(result.max(), 1.0, places=5)
        self.assertEqual(result.dtype, np.float32)

    def test_apply_hu_window_custom_level_width(self):
        vol = np.random.uniform(-500, 500, (5, 32, 32))
        result = apply_hu_window(vol, level=50, width=400)
        self.assertEqual(result.shape, vol.shape)
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)

    def test_apply_hu_window_unknown_fallback(self):
        vol = np.zeros((5, 32, 32))
        result = apply_hu_window(vol, window_name="unknown")
        self.assertEqual(result.shape, vol.shape)

    def test_hu_window_cpu(self):
        img = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
        result = hu_window_cpu(img, low=-100, high=400)
        self.assertEqual(result.shape, (64, 64))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)

    def test_hu_window_cpu_clipping(self):
        img = np.ones((32, 32), dtype=np.float32)
        result = hu_window_cpu(img, low=-100, high=400)
        self.assertAlmostEqual(result.max(), 1.0, places=5)

    def test_hu_window_batch(self):
        batch = torch.rand(4, 64, 64)
        result = hu_window_batch(batch, low=-100, high=400)
        self.assertEqual(result.shape, (4, 64, 64))
        self.assertTrue(result.min() >= -0.01)
        self.assertTrue(result.max() <= 1.01)


class TestNormalization(unittest.TestCase):
    def test_normalize_zscore(self):
        vol = np.random.normal(0, 1, (10, 32, 32)).astype(np.float32)
        result = normalize_volume(vol, method="zscore")
        self.assertEqual(result.shape, vol.shape)
        self.assertEqual(result.dtype, np.float32)

    def test_normalize_minmax(self):
        vol = np.random.uniform(-200, 250, (5, 32, 32)).astype(np.float32)
        result = normalize_volume(vol, method="minmax")
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)

    def test_normalize_constant_volume(self):
        vol = np.full((5, 16, 16), 100.0, dtype=np.float32)
        result_z = normalize_volume(vol, method="zscore")
        self.assertEqual(result_z.shape, vol.shape)

    def test_normalize_invalid_method(self):
        vol = np.zeros((5, 16, 16))
        with self.assertRaises(ValueError):
            normalize_volume(vol, method="invalid")


class TestResize(unittest.TestCase):
    def test_resize_image(self):
        img = np.random.uniform(0, 1, (512, 512)).astype(np.float32)
        result = resize_image(img, (256, 256))
        self.assertEqual(result.shape, (256, 256))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)

    def test_resize_mask_preserves_binary(self):
        mask = np.random.randint(0, 2, (512, 512)).astype(np.float32)
        result = resize_mask(mask, (256, 256))
        self.assertEqual(result.shape, (256, 256))
        unique = np.unique(result)
        for v in unique:
            self.assertIn(v, [0.0, 1.0])


class TestCLAHE(unittest.TestCase):
    def setUp(self):
        self.clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))

    def test_apply(self):
        img = np.random.uniform(0, 1, (256, 256)).astype(np.float32)
        result = self.clahe.apply(img)
        self.assertEqual(result.shape, (256, 256))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.min() >= 0.0)
        self.assertTrue(result.max() <= 1.0)

    def test_apply_batch(self):
        batch = torch.rand(4, 256, 256)
        result = self.clahe.apply_batch(batch)
        self.assertEqual(result.shape, (4, 256, 256))

    def test_serialization(self):
        import pickle
        serialized = pickle.dumps(self.clahe)
        deserialized = pickle.loads(serialized)
        self.assertEqual(deserialized.clip, 2.0)
        self.assertEqual(deserialized.grid, (8, 8))


class TestTransforms(unittest.TestCase):
    def test_preprocessing_transform(self):
        transform = PreprocessingTransform(target_size=(256, 256), hu_low=-100, hu_high=400)
        img = np.random.uniform(0, 1, (512, 512)).astype(np.float32)
        mask = np.random.randint(0, 2, (512, 512)).astype(np.float32)
        img_out, mask_out = transform(img, mask)
        self.assertEqual(img_out.shape, (256, 256))
        self.assertEqual(mask_out.shape, (256, 256))

    def test_preprocessing_with_clahe(self):
        clahe = CLAHEProcessor(clip=2.0, grid=(8, 8))
        transform = PreprocessingTransform(target_size=(256, 256), clahe=clahe)
        img = np.random.uniform(0, 1, (512, 512)).astype(np.float32)
        mask = np.zeros((512, 512), dtype=np.float32)
        img_out, mask_out = transform(img, mask)
        self.assertEqual(img_out.shape, (256, 256))

    def test_augmented_transform_applies_flip(self):
        transform = AugmentedPreprocessingTransform(flip_prob=1.0, shift_range=0.0)
        img = np.tile(np.arange(512, dtype=np.float32).reshape(1, -1), (512, 1)) / 511.0
        mask = np.zeros((512, 512), dtype=np.float32)
        img_out, mask_out = transform(img, mask)
        self.assertEqual(img_out.shape, (256, 256))

    def test_augmented_transform_intensity_shift(self):
        transform = AugmentedPreprocessingTransform(flip_prob=0.0, shift_range=0.5)
        img = np.ones((512, 512), dtype=np.float32) * 0.5
        mask = np.zeros((512, 512), dtype=np.float32)
        img_out, _ = transform(img, mask)
        self.assertEqual(img_out.shape, (256, 256))
        self.assertTrue(img_out.max() <= 1.0)


class TestPatchExtraction(unittest.TestCase):
    def test_extract_patches_3d_basic(self):
        vol = np.random.rand(16, 64, 64).astype(np.float32)
        mask = np.zeros((16, 64, 64), dtype=np.int32)
        mask[4:8, 20:40, 20:40] = 2
        vol_patches, mask_patches = extract_patches_3d(vol, mask, patch_size=(8, 32, 32), stride=(8, 32, 32))
        self.assertEqual(len(vol_patches), len(mask_patches))
        for vp, mp in zip(vol_patches, mask_patches):
            self.assertEqual(vp.shape, (8, 32, 32))
            self.assertEqual(mp.shape, (8, 32, 32))

    def test_extract_patches_balanced(self):
        vol = np.random.rand(16, 64, 64).astype(np.float32)
        mask = np.zeros((16, 64, 64), dtype=np.int32)
        mask[4:8, 20:40, 20:40] = 2
        vol_patches, mask_patches = extract_patches_balanced(
            vol, mask, patch_size=(8, 32, 32), num_patches=20, tumor_ratio=0.5
        )
        total = len(vol_patches)
        self.assertEqual(len(vol_patches), len(mask_patches))
        self.assertLessEqual(total, 20)


class TestClassWeights(unittest.TestCase):
    def test_compute_class_weights(self):
        mask = np.zeros((16, 64, 64), dtype=np.int32)
        mask[5:10, 20:40, 30:50] = 1
        mask[7:9, 25:35, 35:45] = 2
        weights = compute_class_weights(mask)
        self.assertEqual(weights.shape, (3,))
        self.assertIsInstance(weights, torch.Tensor)

    def test_one_hot_encode(self):
        mask = torch.randint(0, 3, (2, 8, 16, 16))
        one_hot = one_hot_encode(mask, num_classes=3)
        self.assertEqual(one_hot.shape, (2, 3, 8, 16, 16))


class TestGPUBatchPreprocessing(unittest.TestCase):
    def test_preprocess_batch_gpu(self):
        images = torch.rand(2, 512, 512)
        masks = torch.randint(0, 2, (2, 512, 512)).float()
        imgs_out, masks_out = preprocess_batch_gpu(images, masks, (256, 256))
        self.assertEqual(imgs_out.shape, (2, 256, 256))
        self.assertEqual(masks_out.shape, (2, 256, 256))


if __name__ == "__main__":
    unittest.main()

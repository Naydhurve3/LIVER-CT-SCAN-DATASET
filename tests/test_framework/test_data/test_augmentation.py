import unittest
import torch
from src.framework.data.augmentations import (
    RandomFlip3D, RandomRotate90, RandomIntensityShift,
    RandomNoise, RandomElasticDeformation3D, Compose3D,
)


class TestRandomFlip3D(unittest.TestCase):
    def setUp(self):
        self.image = torch.rand(1, 8, 32, 32)
        self.mask = torch.randint(0, 2, (8, 32, 32))

    def test_flip_always_applied(self):
        transform = RandomFlip3D(axes=(1, 2), p=1.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertEqual(result["mask"].shape, self.mask.shape)

    def test_flip_never_applied(self):
        transform = RandomFlip3D(axes=(1, 2), p=0.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertTrue(torch.allclose(result["image"], self.image))
        self.assertTrue(torch.allclose(result["mask"].float(), self.mask.float()))

    def test_flip_preserves_dtype(self):
        transform = RandomFlip3D(axes=(1, 2), p=1.0)
        result = transform(self.image, self.mask)
        self.assertEqual(result["image"].dtype, self.image.dtype)
        self.assertEqual(result["mask"].dtype, self.mask.dtype)


class TestRandomRotate90(unittest.TestCase):
    def setUp(self):
        self.image = torch.rand(1, 8, 32, 32)
        self.mask = torch.randint(0, 2, (8, 32, 32))

    def test_rotate_always_applied(self):
        transform = RandomRotate90(p=1.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertEqual(result["mask"].shape, self.mask.shape)

    def test_rotate_never_applied(self):
        transform = RandomRotate90(p=0.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertTrue(torch.allclose(result["image"], self.image))
        self.assertTrue(torch.allclose(result["mask"].float(), self.mask.float()))


class TestRandomIntensityShift(unittest.TestCase):
    def setUp(self):
        self.image = torch.full((1, 4, 16, 16), 0.5)
        self.mask = torch.randint(0, 2, (4, 16, 16))

    def test_intensity_shift_always_applied(self):
        transform = RandomIntensityShift(shift_range=0.2, scale_range=0.0, p=1.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertFalse(torch.allclose(result["image"], self.image))

    def test_intensity_shift_never_applied(self):
        transform = RandomIntensityShift(shift_range=0.2, scale_range=0.0, p=0.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertTrue(torch.allclose(result["image"], self.image))


class TestRandomNoise(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros(1, 4, 16, 16)
        self.mask = torch.zeros(4, 16, 16, dtype=torch.long)

    def test_noise_always_applied(self):
        transform = RandomNoise(noise_std=0.1, p=1.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertFalse(torch.allclose(result["image"], self.image))

    def test_noise_never_applied(self):
        transform = RandomNoise(noise_std=0.1, p=0.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertTrue(torch.allclose(result["image"], self.image))


class TestRandomElasticDeformation3D(unittest.TestCase):
    def setUp(self):
        self.image = torch.rand(1, 8, 32, 32)
        self.mask = torch.randint(0, 2, (8, 32, 32))

    def test_elastic_always_applied(self):
        transform = RandomElasticDeformation3D(alpha=10, sigma=3, p=1.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertEqual(result["mask"].shape, self.mask.shape)

    def test_elastic_never_applied(self):
        transform = RandomElasticDeformation3D(alpha=10, sigma=3, p=0.0)
        result = transform(self.image.clone(), self.mask.clone())
        self.assertTrue(torch.allclose(result["image"], self.image))
        self.assertTrue(torch.allclose(result["mask"].float(), self.mask.float()))


class TestCompose3D(unittest.TestCase):
    def setUp(self):
        self.image = torch.rand(1, 8, 32, 32)
        self.mask = torch.randint(0, 2, (8, 32, 32))

    def test_compose_applies_all(self):
        transforms = [
            RandomFlip3D(axes=(1, 2), p=1.0),
            RandomRotate90(p=1.0),
            RandomIntensityShift(shift_range=0.1, p=1.0),
        ]
        compose = Compose3D(transforms)
        result = compose(self.image.clone(), self.mask.clone())
        self.assertEqual(result["image"].shape, self.image.shape)
        self.assertEqual(result["mask"].shape, self.mask.shape)

    def test_compose_no_transforms(self):
        compose = Compose3D([])
        result = compose(self.image.clone(), self.mask.clone())
        self.assertTrue(torch.allclose(result["image"], self.image))
        self.assertTrue(torch.allclose(result["mask"].float(), self.mask.float()))

    def test_compose_preserves_dtype(self):
        compose = Compose3D([RandomFlip3D(axes=(1, 2), p=1.0)])
        result = compose(self.image, self.mask)
        self.assertEqual(result["image"].dtype, self.image.dtype)
        self.assertEqual(result["mask"].dtype, self.mask.dtype)


if __name__ == "__main__":
    unittest.main()

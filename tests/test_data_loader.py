import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from src.data_loader import (
    DatasetConfig, DataPathManager, VolumeWiseSplitter,
    LiverTumor2DDataset, create_2d_dataloaders,
)


class TestDatasetConfig(unittest.TestCase):
    def test_paths_are_valid(self):
        self.assertIsInstance(DatasetConfig.PROJECT_DIR, Path)
        self.assertTrue(str(DatasetConfig.PROJECT_DIR).endswith("Liver"))

    def test_output_dirs_exist(self):
        DatasetConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.assertTrue(DatasetConfig.OUTPUT_DIR.exists())
        DatasetConfig.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        self.assertTrue(DatasetConfig.SPLITS_DIR.exists())


class TestDataPathManager(unittest.TestCase):
    def setUp(self):
        self.manager = DataPathManager()

    def test_extract_volume_id_image(self):
        vid = self.manager.extract_volume_id("Volume-005-012.png")
        self.assertEqual(vid, 5)

    def test_extract_volume_id_mask(self):
        vid = self.manager.extract_volume_id("mask-123-045.png")
        self.assertEqual(vid, 123)

    def test_extract_volume_id_invalid(self):
        vid = self.manager.extract_volume_id("random.txt")
        self.assertIsNone(vid)

    def test_extract_slice_id(self):
        sid = self.manager.extract_slice_id("Volume-005-012.png")
        self.assertEqual(sid, 12)

    def test_extract_slice_id_no_match(self):
        sid = self.manager.extract_slice_id("random.txt")
        self.assertEqual(sid, 0)


class TestVolumeWiseSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = VolumeWiseSplitter(split_ratios=(0.8, 0.1, 0.1))

    def test_split_returns_all_ids(self):
        volume_ids = list(range(100))
        splits = self.splitter.split(volume_ids)
        all_ids = splits['train'] + splits['val'] + splits['test']
        self.assertEqual(sorted(all_ids), volume_ids)
        self.assertEqual(len(splits['train']), 80)
        self.assertEqual(len(splits['val']), 10)
        self.assertEqual(len(splits['test']), 10)

    def test_split_no_overlap(self):
        volume_ids = list(range(50))
        splits = self.splitter.split(volume_ids)
        train_set = set(splits['train'])
        val_set = set(splits['val'])
        test_set = set(splits['test'])
        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(val_set.isdisjoint(test_set))

    def test_save_and_load_splits_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            volume_ids = list(range(30))
            splits = self.splitter.split(volume_ids)
            self.splitter.save_splits(splits, tmp_path)
            loaded = self.splitter.load_splits(tmp_path)
            self.assertEqual(set(loaded['train']), set(splits['train']))
            self.assertEqual(set(loaded['val']), set(splits['val']))
            self.assertEqual(set(loaded['test']), set(splits['test']))


class TestLiverTumor2DDataset(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.img_dir = Path(self.tmp_dir.name) / "images"
        self.mask_dir = Path(self.tmp_dir.name) / "masks"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        for vid in range(2):
            for sid in range(3):
                img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
                img.save(self.img_dir / f"Volume-{vid:03d}-{sid:03d}.png")
                mask = Image.fromarray(np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255)
                mask.save(self.mask_dir / f"mask-{vid:03d}-{sid:03d}.png")

        self.volume_index = {
            'image_paths': {
                v: sorted(self.img_dir.glob(f"Volume-{v:03d}-*.png"))
                for v in range(2)
            },
            'mask_paths': {
                v: sorted(self.mask_dir.glob(f"mask-{v:03d}-*.png"))
                for v in range(2)
            },
            'volumes': [0, 1],
        }

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_dataset_len(self):
        dataset = LiverTumor2DDataset([0, 1], self.volume_index)
        self.assertEqual(len(dataset), 6)

    def test_dataset_getitem(self):
        dataset = LiverTumor2DDataset([0], self.volume_index)
        item = dataset[0]
        self.assertIn('image', item)
        self.assertIn('mask', item)
        self.assertIn('volume_id', item)
        self.assertIn('slice_id', item)
        self.assertEqual(item['image'].shape, (1, 64, 64))
        self.assertEqual(item['mask'].shape, (1, 64, 64))
        self.assertEqual(item['volume_id'], 0)
        self.assertEqual(item['slice_id'], 0)

    def test_dataset_no_mask_fallback(self):
        index = {
            'image_paths': {0: self.volume_index['image_paths'][0]},
            'mask_paths': {},
            'volumes': [0],
        }
        dataset = LiverTumor2DDataset([0], index)
        item = dataset[0]
        self.assertEqual(item['mask'].shape, (1, 64, 64))
        self.assertTrue((item['mask'] == 0).all())


class TestCreateDataloaders(unittest.TestCase):
    def test_create_dataloaders_basic(self):
        volume_index = {
            'image_paths': {v: [] for v in range(3)},
            'mask_paths': {v: [] for v in range(3)},
            'volumes': [0, 1, 2],
        }
        train_loader, val_loader, test_loader = create_2d_dataloaders(
            volume_index, [0], [1], [2],
            batch_size=2, num_workers=0,
        )
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)


if __name__ == "__main__":
    unittest.main()

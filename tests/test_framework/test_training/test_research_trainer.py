import torch
from torch.utils.data import DataLoader, Dataset

from src.framework.losses.focal_dice import FocalDiceLoss
from src.framework.training.research_trainer import ResearchTrainer


class TinyDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.zeros(1, 8, 8)
        mask = torch.zeros(1, 8, 8)
        image[:, 2:6, 2:6] = 1
        mask[:, 2:6, 2:6] = 1
        return {"image": image, "mask": mask, "volume_id": 1, "slice_id": index}


def test_synthetic_one_epoch_checkpoint_reload(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Conv2d(1, 1, 1)
    loader = DataLoader(TinyDataset(), batch_size=2)
    trainer = ResearchTrainer(
        model, FocalDiceLoss(), loader, loader, tmp_path,
        epochs=1, mixed_precision=False, device=torch.device("cpu"),
    )
    result = trainer.fit()
    assert (tmp_path / "best_checkpoint.pth").exists()
    assert result["best_epoch"] == 1

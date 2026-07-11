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
    assert result["status"] == "completed"


def test_resume_continues_to_total_target_epoch(tmp_path):
    torch.manual_seed(42)
    loader = DataLoader(TinyDataset(), batch_size=2)
    first = ResearchTrainer(
        torch.nn.Conv2d(1, 1, 1), FocalDiceLoss(), loader, loader, tmp_path,
        epochs=1, schedule_epochs=3, mixed_precision=False, device=torch.device("cpu"),
    )
    first.fit()
    resumed = ResearchTrainer(
        torch.nn.Conv2d(1, 1, 1), FocalDiceLoss(), loader, loader, tmp_path,
        epochs=3, schedule_epochs=3, mixed_precision=False, device=torch.device("cpu"),
    )
    info = resumed.resume(tmp_path / "last_checkpoint.pth")
    assert info["start_epoch"] == 1
    assert info["resume_reproducibility"] == "exact"
    result = resumed.fit()
    assert result["completed_epochs"] == 3
    assert [row["epoch"] for row in result["history"]] == [1, 2, 3]


def test_interrupt_saves_epoch_start_checkpoint(tmp_path):
    loader = DataLoader(TinyDataset(), batch_size=2)
    trainer = ResearchTrainer(
        torch.nn.Conv2d(1, 1, 1), FocalDiceLoss(), loader, loader, tmp_path,
        epochs=2, mixed_precision=False, device=torch.device("cpu"),
    )

    def interrupt(_epoch):
        raise KeyboardInterrupt

    trainer.train_epoch = interrupt
    result = trainer.fit()
    assert result["status"] == "interrupted"
    assert (tmp_path / "interrupted_checkpoint.pth").exists()
    payload = torch.load(tmp_path / "interrupted_checkpoint.pth", weights_only=True)
    assert payload["epoch"] == 0


def test_legacy_resume_infers_best_score_and_scheduler(tmp_path):
    loader = DataLoader(TinyDataset(), batch_size=2)
    source = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    legacy = {
        "model_state": source.state_dict(), "optimizer_state": optimizer.state_dict(),
        "epoch": 1, "best_positive_volume_dice": float("-inf"),
        "history": [{"epoch": 1, "train_loss": 0.5,
                     "val_positive_volume_dice_at_0_5": 0.2, "lr": 0.000975528}],
    }
    path = tmp_path / "legacy.pth"
    torch.save(legacy, path)
    trainer = ResearchTrainer(
        torch.nn.Conv2d(1, 1, 1), FocalDiceLoss(), loader, loader, tmp_path,
        epochs=5, schedule_epochs=10, mixed_precision=False, device=torch.device("cpu"),
    )
    info = trainer.resume(path)
    assert info["best_score"] == 0.2
    assert info["best_epoch"] == 1
    assert info["resume_reproducibility"].startswith("legacy_scheduler_reconstructed")

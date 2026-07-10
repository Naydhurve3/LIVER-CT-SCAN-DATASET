from pathlib import Path

import torch

from src.framework.evaluation.checkpoints import extract_state_dict, load_checkpoint_into_model


def test_extract_raw_state_dict():
    model = torch.nn.Linear(2, 1)
    state, metadata = extract_state_dict(model.state_dict())
    assert "weight" in state
    assert metadata == {}


def test_load_trainer_checkpoint(tmp_path):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state": source.state_dict(), "epoch": 3}, path)
    target = torch.nn.Linear(2, 1)
    info = load_checkpoint_into_model(target, path)
    assert info["metadata"]["epoch"] == 3
    assert torch.equal(source.weight, target.weight)


def test_load_raw_checkpoint(tmp_path):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "raw.pth"
    torch.save(source.state_dict(), path)
    target = torch.nn.Linear(2, 1)
    load_checkpoint_into_model(target, path)
    assert torch.equal(source.bias, target.bias)

from tools.ablate import ABLATIONS, run_single_ablation


def test_ablation_configs_defined():
    assert len(ABLATIONS) >= 8
    names = [c["name"] for c in ABLATIONS]
    assert "baseline" in names
    assert "faupnet" in names
    assert "faupnet_uwaclv2" in names


def test_ablation_configs_have_required_keys():
    for cfg in ABLATIONS:
        assert "name" in cfg
        assert "model" in cfg
        assert "loss" in cfg
        assert "model" in cfg
        assert "in_channels" in cfg["model"]
        assert "out_channels" in cfg["model"]


def test_ablation_names_are_unique():
    names = [c["name"] for c in ABLATIONS]
    assert len(names) == len(set(names))


def test_ablation_zero_shot_returns_metrics():
    import torch
    from src.framework.core.factory import build_model
    from src.framework.data.transforms import PreprocessingTransform
    from src.framework.data.lits_dataset import DataPathManager, VolumeWiseSplitter, create_2d_dataloaders
    from src.framework.utils.gpu_utils import DEVICE

    path_manager = DataPathManager(".", ".")
    volume_index = path_manager.build_index()
    if not volume_index.get("volumes"):
        return

    transform = PreprocessingTransform(target_size=(256, 256))
    _, val_loader, _ = create_2d_dataloaders(volume_index, [], volume_index["volumes"][:2], [],
                                              batch_size=2, transform_val=transform)
    result = run_single_ablation(ABLATIONS[0], val_loader)
    assert "dice" in result
    assert "iou" in result

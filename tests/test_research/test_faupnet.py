import torch
from src.research.faupnet.model import FAUPNet, GatedSkipConnection, UncertaintyHead


def test_faupnet_forward():
    model = FAUPNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    assert y.shape == (2, 1, 256, 256)


def test_faupnet_predict():
    model = FAUPNet(in_channels=1, out_channels=1)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        p = model.predict(x)
    assert p.shape == (2, 1, 256, 256)
    assert p.min() >= 0 and p.max() <= 1


def test_faupnet_uncertainty_maps():
    model = FAUPNet(in_channels=1, out_channels=1, gate_levels=[3, 4])
    x = torch.randn(2, 1, 256, 256)
    _ = model(x)
    maps = model.get_uncertainty_maps()
    assert "gate_level_3" in maps or "gate_level_4" in maps


def test_uncertainty_head():
    head = UncertaintyHead(32, hidden=8)
    x = torch.randn(2, 32, 64, 64)
    u = head(x)
    assert u.shape == (2, 1, 64, 64)


def test_gated_skip():
    gate = GatedSkipConnection(32)
    enc = torch.randn(2, 32, 64, 64)
    out, u = gate(enc)
    assert out.shape[1] == 32
    assert u.shape == (2, 1, 64, 64)


def test_faupnet_registered():
    from src.framework.core.registry import MODELS
    assert "faupnet" in MODELS


def test_faupnet_different_gate_levels():
    model = FAUPNet(in_channels=1, out_channels=1, gate_levels=[2, 4])
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    assert y.shape == (1, 1, 256, 256)

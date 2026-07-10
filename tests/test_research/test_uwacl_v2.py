import torch
from src.research.uwacl.loss_v2 import UWACLv2MultiScale
from src.framework.core.registry import LOSSES


def test_uwacl_v2_forward():
    loss_fn = UWACLv2MultiScale(beta=5.0, tau=0.1)
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    l = loss_fn(pred, target)
    assert l > 0
    assert torch.isfinite(l)


def test_uwacl_v2_with_uncertainty():
    loss_fn = UWACLv2MultiScale(beta=5.0, tau=0.1)
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    uncertainty = torch.rand(2, 1, 64, 64) * 0.3
    l = loss_fn(pred, target, uncertainty)
    assert l > 0
    assert torch.isfinite(l)


def test_uwacl_v2_tau_schedule():
    loss_fn = UWACLv2MultiScale(tau=0.1, tau_min=0.01)
    loss_fn.set_epoch(25, 50)
    assert loss_fn._current_tau < 0.1
    assert loss_fn._current_tau >= 0.01


def test_uwacl_v2_multi_scale_uncertainty():
    loss_fn = UWACLv2MultiScale()
    u = torch.rand(2, 1, 64, 64)
    u_multi = loss_fn._multi_scale_uncertainty(u)
    assert u_multi.shape == u.shape


def test_uwacl_v2_edge_loss():
    loss_fn = UWACLv2MultiScale(edge_weight=0.1)
    pred = torch.randn(2, 1, 64, 64)
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    l = loss_fn(pred, target)
    assert torch.isfinite(l)


def test_uwacl_v2_set_tau():
    loss_fn = UWACLv2MultiScale(tau=0.1)
    loss_fn.set_tau(0.05)
    assert loss_fn._current_tau == 0.05


def test_uwacl_v2_registered():
    assert "uwacl_v2_multi" in LOSSES


def test_uwacl_v2_no_uncertainty_edge_only():
    loss_fn = UWACLv2MultiScale(edge_weight=0.5)
    pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    l = loss_fn(pred, target)
    assert torch.isfinite(l)

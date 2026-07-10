import random
import numpy as np
import torch
from src.framework.core.reproducibility import set_seed


def test_set_seed_deterministic():
    set_seed(42, deterministic=True)
    a = random.random()
    b = np.random.randn(5)
    c = torch.randn(5)
    set_seed(42, deterministic=True)
    assert a == random.random()
    assert np.allclose(b, np.random.randn(5))
    assert torch.allclose(c, torch.randn(5))

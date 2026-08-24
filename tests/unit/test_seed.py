"""Unit tests for deterministic pseudo-random number generator seeding."""

import random
import numpy as np
from src.utils.seed import set_seed

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_seed_determinism():
    """Verify that setting the seed produces reproducible pseudo-random outputs."""
    set_seed(42)
    py_val_1 = random.random()
    np_val_1 = np.random.rand()

    set_seed(42)
    py_val_2 = random.random()
    np_val_2 = np.random.rand()

    assert py_val_1 == py_val_2
    assert np_val_1 == np_val_2

    if HAS_TORCH:
        set_seed(42)
        th_val_1 = torch.rand(1).item()
        set_seed(42)
        th_val_2 = torch.rand(1).item()
        assert th_val_1 == th_val_2

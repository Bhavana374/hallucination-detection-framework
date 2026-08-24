"""Reproducibility utilities for deterministic pseudo-random number generator seeding."""

import os
import random
from typing import Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch backends.

    Args:
        seed: Integer seed value.
        deterministic_cuda: If True, configures CuDNN backends for strict determinism.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if NUMPY_AVAILABLE:
        np.random.seed(seed)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            if deterministic_cuda:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False


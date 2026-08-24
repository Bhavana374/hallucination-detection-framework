"""Hardware device detection and memory inspection utility for PyTorch."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device(preferred_device: str = "auto") -> Any:
    """Resolve and return an optimal device instance (torch.device if PyTorch is installed, else string).

    Args:
        preferred_device: 'auto', 'cuda', 'cpu', or 'mps'.

    Returns:
        torch.device instance or string fallback.
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch is not installed. Returning device string descriptor.")
        return "cpu"

    pref = preferred_device.lower().strip()

    if pref == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif pref == "cpu":
        device = torch.device("cpu")
    else:  # 'auto'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device


def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed hardware accelerator specifications and memory status.

    Returns:
        Dictionary containing accelerator metadata.
    """
    if not TORCH_AVAILABLE:
        return {
            "cuda_available": False,
            "device_count": 0,
            "torch_version": "NOT_INSTALLED",
            "current_device_name": "CPU (PyTorch not installed)",
            "total_memory_gb": None,
        }

    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        info["current_device_name"] = props.name
        info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        info["allocated_memory_mb"] = round(torch.cuda.memory_allocated(current_idx) / (1024 ** 2), 2)
        info["reserved_memory_mb"] = round(torch.cuda.memory_reserved(current_idx) / (1024 ** 2), 2)
    else:
        info["current_device_name"] = "CPU"
        info["total_memory_gb"] = None

    return info


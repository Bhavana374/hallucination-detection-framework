"""Unit tests for PyTorch device resolution and hardware inspection."""

from src.utils.device import get_device, get_device_info

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_get_device_cpu():
    """Test explicit CPU device request."""
    dev = get_device("cpu")
    if HAS_TORCH:
        assert dev.type == "cpu"
    else:
        assert dev == "cpu"


def test_get_device_auto():
    """Test automatic device resolution."""
    dev = get_device("auto")
    if HAS_TORCH:
        assert isinstance(dev, torch.device)
        assert dev.type in ("cuda", "cpu", "mps")
    else:
        assert dev in ("cuda", "cpu", "mps")


def test_get_device_info():
    """Test retrieving device hardware specifications."""
    info = get_device_info()
    assert "cuda_available" in info
    assert "device_count" in info
    assert "torch_version" in info
    assert "current_device_name" in info

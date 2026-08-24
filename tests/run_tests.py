"""
Standard library test runner for verifying environment, modules, and full framework pipeline.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestEnvironmentAndUtils(unittest.TestCase):
    """Test environment loading, core utilities, and project structure."""

    def test_imports(self):
        """Test importing core project packages."""
        import src
        import src.utils
        import src.utils.logger
        import src.utils.config
        import src.utils.device
        import src.utils.seed
        import src.data
        import src.preprocessing
        import src.retrieval
        import src.features
        import src.models
        import src.models.baselines
        import src.models.bert
        import src.models.deberta
        import src.models.hybrid
        import src.training
        import src.evaluation
        import src.explainability
        import src.pipeline
        import api
        import api.routes
        import api.schemas

        self.assertIsNotNone(src.__version__)
        print("[PASS] All project packages and submodules imported successfully.")

    def test_logger(self):
        """Test logger utility."""
        from src.utils.logger import setup_logger
        logger = setup_logger("test_unit", log_level="DEBUG", log_file=None, console_output=False)
        self.assertEqual(logger.name, "test_unit")
        print("[PASS] Logger utility initialized successfully.")

    def test_config_dict(self):
        """Test ConfigDict dot-access and dictionary merging."""
        from src.utils.config import ConfigDict, merge_dicts
        base = {"a": 1, "nested": {"x": 10, "y": 20}}
        override = {"b": 2, "nested": {"y": 99}}
        merged = merge_dicts(base, override)
        cfg = ConfigDict(merged)
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, 2)
        self.assertEqual(cfg.nested.y, 99)
        print("[PASS] ConfigDict and dictionary merge tested successfully.")

    def test_device(self):
        """Test device utility."""
        from src.utils.device import get_device, get_device_info
        dev = get_device("cpu")
        info = get_device_info()
        self.assertIsNotNone(dev)
        self.assertIn("cuda_available", info)
        print(f"[PASS] Device utility resolved device successfully: {dev}")

    def test_seed(self):
        """Test deterministic seed utility."""
        import random
        from src.utils.seed import set_seed
        set_seed(42)
        val1 = random.random()
        set_seed(42)
        val2 = random.random()
        self.assertEqual(val1, val2)
        print("[PASS] Deterministic seed utility verified.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE FRAMEWORK TEST SUITE")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

"""Unit test verifying that core modules and configurations import seamlessly."""

import importlib
import pytest


def test_src_imports():
    """Verify that all core src subpackages import cleanly."""
    modules = [
        "src",
        "src.data",
        "src.preprocessing",
        "src.retrieval",
        "src.features",
        "src.models",
        "src.models.baselines",
        "src.models.bert",
        "src.models.deberta",
        "src.models.hybrid",
        "src.training",
        "src.evaluation",
        "src.explainability",
        "src.pipeline",
        "src.utils",
        "src.utils.logger",
        "src.utils.config",
        "src.utils.device",
        "src.utils.seed",
    ]
    for mod in modules:
        imported = importlib.import_module(mod)
        assert imported is not None


def test_api_imports():
    """Verify that api package and subpackages import cleanly."""
    modules = [
        "api",
        "api.routes",
        "api.schemas",
    ]
    for mod in modules:
        imported = importlib.import_module(mod)
        assert imported is not None

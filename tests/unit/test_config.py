"""Unit tests for configuration utilities and YAML loading."""

import pytest
from pathlib import Path
from src.utils.config import load_yaml, load_all_configs, merge_dicts, ConfigDict


def test_load_valid_yaml(tmp_path: Path):
    """Test loading a properly formatted YAML file."""
    yaml_content = """
    project:
      name: "Test Project"
      seed: 42
    models:
      - bert
      - deberta
    """
    test_file = tmp_path / "test_config.yaml"
    test_file.write_text(yaml_content, encoding="utf-8")

    loaded = load_yaml(test_file)
    assert loaded["project"]["name"] == "Test Project"
    assert loaded["project"]["seed"] == 42
    assert "bert" in loaded["models"]


def test_load_nonexistent_yaml():
    """Test error handling when loading a missing YAML file."""
    with pytest.raises(FileNotFoundError):
        load_yaml("nonexistent_path/config.yaml")


def test_merge_dicts():
    """Test recursive dictionary merging."""
    base = {"a": 1, "nested": {"x": 10, "y": 20}}
    override = {"b": 2, "nested": {"y": 99, "z": 30}}
    merged = merge_dicts(base, override)

    assert merged["a"] == 1
    assert merged["b"] == 2
    assert merged["nested"]["x"] == 10
    assert merged["nested"]["y"] == 99
    assert merged["nested"]["z"] == 30


def test_config_dict_dot_access():
    """Test dot-notation access on ConfigDict."""
    raw_dict = {"project": {"name": "Hallucination Framework", "seed": 42}}
    cfg = ConfigDict(raw_dict)

    assert cfg.project.name == "Hallucination Framework"
    assert cfg.project.seed == 42


def test_load_all_configs():
    """Test loading the project's actual configuration files."""
    cfg = load_all_configs(configs_dir="configs")
    assert "project" in cfg
    assert "dataset" in cfg
    assert "baselines" in cfg
    assert "retrieval" in cfg
    assert "experiments" in cfg
    assert cfg.project.random_seed == 42

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False



class ConfigDict(dict):
    """Dictionary subclass providing dot-notation attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Configuration has no parameter '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no parameter '{key}'")


def _simple_yaml_fallback(filepath: Path) -> Dict[str, Any]:
    """Lightweight fallback parser for basic YAML key-value pairs when PyYAML is unavailable."""
    result: Dict[str, Any] = {}
    current_section: Optional[str] = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.split("#")[0].rstrip()
            if not raw_line:
                continue

            # Check for top-level section: "section:"
            if not raw_line.startswith(" ") and ":" in raw_line and not raw_line.startswith("-"):
                parts = raw_line.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                if val:
                    result[key] = _parse_val(val)
                    current_section = None
                else:
                    result[key] = {}
                    current_section = key
            elif current_section and raw_line.startswith("  ") and ":" in raw_line:
                sub_line = raw_line.strip()
                parts = sub_line.split(":", 1)
                sub_key = parts[0].strip()
                sub_val = parts[1].strip()
                if isinstance(result[current_section], dict):
                    result[current_section][sub_key] = _parse_val(sub_val)

    return result


def _parse_val(val: str) -> Any:
    """Parse string representation of primitive types."""
    v = val.strip().strip('"').strip("'")
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.isdigit():
        return int(v)
    try:
        return float(v)
    except ValueError:
        return v


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML file safely.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    if YAML_AVAILABLE:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    else:
        return _simple_yaml_fallback(path)




def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_dicts(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_all_configs(configs_dir: Union[str, Path] = "configs") -> ConfigDict:
    """Load and combine standard configuration YAML files from configs directory.

    Loads:
      - config.yaml (global settings & paths)
      - data.yaml (dataset and preprocessing)
      - model.yaml (model architectures & hyperparameters)
      - retrieval.yaml (vector indexing and retrieval)
      - experiment.yaml (experiment tracking and ablations)

    Args:
        configs_dir: Directory containing YAML config files.

    Returns:
        ConfigDict with consolidated configuration accessible by dot notation.
    """
    dir_path = Path(configs_dir)
    config_files = ["config.yaml", "data.yaml", "model.yaml", "retrieval.yaml", "experiment.yaml"]
    combined_config: Dict[str, Any] = {}

    for file_name in config_files:
        file_path = dir_path / file_name
        if file_path.is_file():
            data = load_yaml(file_path)
            combined_config = merge_dicts(combined_config, data)

    return ConfigDict(combined_config)

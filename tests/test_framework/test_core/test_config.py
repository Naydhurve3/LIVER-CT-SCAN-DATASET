import pytest
import tempfile
from pathlib import Path
from src.framework.core.config import load_config, merge_configs
from src.framework.core.exceptions import ConfigError


def test_load_config_yaml():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("key: value\nnested:\n  inner: 42\n")
        path = f.name
    config = load_config(path)
    assert config["key"] == "value"
    assert config["nested"]["inner"] == 42
    Path(path).unlink()


def test_merge_configs_deep():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 4}
    merged = merge_configs(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 99
    assert merged["b"]["d"] == 3
    assert merged["e"] == 4


def test_merge_configs_scalar_overrides():
    base = {"training": {"batch_size": 8, "epochs": 50, "mixed_precision": True}}
    override = {"training": {"batch_size": 4}}
    merged = merge_configs(base, override)
    assert merged["training"]["batch_size"] == 4
    assert merged["training"]["epochs"] == 50
    assert merged["training"]["mixed_precision"] is True


def test_load_config_file_not_found():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.yaml")

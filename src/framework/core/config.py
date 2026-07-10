import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.framework.core.exceptions import ConfigError


def _resolve_env_ref(value: str) -> str:
    if value.startswith("${env:") and value.endswith("}"):
        env_var = value[6:-1]
        resolved = os.getenv(env_var)
        if resolved is None:
            raise ConfigError(f"Environment variable '{env_var}' is not set")
        return resolved
    return value


def _resolve_refs(obj: Any) -> Any:
    if isinstance(obj, str):
        return _resolve_env_ref(obj)
    elif isinstance(obj, dict):
        return {k: _resolve_refs(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_refs(v) for v in obj]
    return obj


def load_config(path: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return _resolve_refs(config)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def resolve_inherits(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    inherits = config.pop("inherits", [])
    merged: Dict[str, Any] = {}
    for ref in inherits:
        ref_path = config_dir / f"{ref}.yaml"
        ref_config = load_config(str(ref_path))
        ref_config = resolve_inherits(ref_config, ref_path.parent)
        merged = merge_configs(merged, ref_config)
    merged = merge_configs(merged, config)
    return merged


def build_experiment_config(experiment_path: str) -> Dict[str, Any]:
    experiment_path = Path(experiment_path)
    config_dir = experiment_path.parent
    raw = load_config(str(experiment_path))
    resolved = resolve_inherits(raw, config_dir)
    return resolved

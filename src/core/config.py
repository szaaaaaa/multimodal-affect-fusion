"""
Hierarchical YAML configuration loader with base + override support.

分层 YAML 配置加载器，支持 base + override 合并。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config(dict):
    """
    Dot-accessible nested dict for configuration.

    支持点号访问的嵌套字典配置。
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten nested config to dot-separated keys."""
        flat: Dict[str, Any] = {}
        _flatten("", self, flat)
        return flat


def _flatten(prefix: str, d: dict, out: dict) -> None:
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            _flatten(key, v, out)
        else:
            out[key] = v


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _apply_dot_overrides(cfg: dict, overrides: List[str]) -> dict:
    """
    Apply CLI overrides like 'model.fusion.name=mult'.

    应用命令行覆盖，如 'model.fusion.name=mult'。
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parts = key.strip().split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        # Try to parse value as JSON (handles lists, ints, floats, bools)
        try:
            d[parts[-1]] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            d[parts[-1]] = value
    return cfg


def load_config(
    path: str | Path,
    overrides: Optional[List[str]] = None,
) -> Config:
    """
    Load a YAML config with optional _base_ inheritance and CLI overrides.

    加载 YAML 配置，支持 _base_ 继承和命令行覆盖。

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.
    overrides : list of str, optional
        CLI overrides in "key.subkey=value" format.

    Returns
    -------
    Config
        Merged configuration.
    """
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Resolve _base_ inheritance
    if "_base_" in raw:
        base_path = (path.parent / raw.pop("_base_")).resolve()
        base_cfg = load_config(base_path)
        raw = _deep_merge(dict(base_cfg), raw)

    # Apply CLI overrides
    if overrides:
        raw = _apply_dot_overrides(raw, overrides)

    return Config(raw)


def save_config(cfg: dict, path: str | Path) -> None:
    """Save config dict to YAML file."""
    if not YAML_AVAILABLE:
        # Fallback to JSON
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(dict(cfg), f, indent=2, ensure_ascii=False, default=str)
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            dict(cfg),
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

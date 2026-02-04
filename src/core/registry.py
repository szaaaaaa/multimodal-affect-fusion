"""
Registry mechanism for plugin-based module discovery and construction.

基于注册表的模块发现与构建机制。

Usage:
    @FUSIONS.register("lft")
    class LFTFusion(BaseFusion):
        def __init__(self, cfg): ...

    fusion = FUSIONS.build("lft", cfg)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    """
    Generic string-keyed registry.

    通用字符串键注册表。
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}

    def register(self, key: str) -> Callable:
        """Decorator to register a class under *key*."""
        def decorator(cls: Type) -> Type:
            if key in self._registry:
                raise KeyError(
                    f"[{self.name}] '{key}' is already registered "
                    f"(existing: {self._registry[key].__name__})"
                )
            self._registry[key] = cls
            return cls
        return decorator

    def build(self, key: str, cfg: Any = None):
        """Look up *key* and instantiate with *cfg*."""
        if key not in self._registry:
            raise KeyError(
                f"[{self.name}] '{key}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        cls = self._registry[key]
        if cfg is None:
            return cls()
        return cls(cfg)

    def get(self, key: str) -> Type:
        """Return the class registered under *key* without instantiating."""
        if key not in self._registry:
            raise KeyError(
                f"[{self.name}] '{key}' not found. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, keys={list(self._registry.keys())})"


# ──────────────────────────────────────────────
# Global registries
# ──────────────────────────────────────────────

ENCODERS: Dict[str, Registry] = {}
FUSIONS = Registry("fusions")
DATAMODULES = Registry("datamodules")
HEADS = Registry("heads")
LOSSES = Registry("losses")
METRICS = Registry("metrics")


def get_encoder_registry(modality: str) -> Registry:
    """
    Get (or auto-create) the encoder registry for *modality*.

    按模态获取 encoder 注册表，不存在则自动创建。
    """
    if modality not in ENCODERS:
        ENCODERS[modality] = Registry(f"{modality}_encoders")
    return ENCODERS[modality]

from __future__ import annotations

from importlib import import_module

__all__ = ["DeepLOBLite", "TLOBStarter", "IntervalResidualRidge"]

_MODULE_MAP = {
    "DeepLOBLite": "models.deeplob_lite",
    "TLOBStarter": "models.tlob_starter",
    "IntervalResidualRidge": "models.interval_residual",
}


def __getattr__(name):
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_MODULE_MAP[name])
    return getattr(module, name)

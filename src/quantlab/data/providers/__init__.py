"""Market data provider plugins.

Built-ins: ``yahoo`` (free, no API key, via yfinance) and ``csv``. Third-party
providers register under the ``quantlab.data_providers`` entry-point group or are
referenced directly as ``package.module:ClassName``.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from quantlab.data.providers.base import DataProvider

ENTRY_POINT_GROUP = "quantlab.data_providers"
BUILTINS = {
    "yahoo": "quantlab.data.providers.yahoo:YahooProvider",
    "csv": "quantlab.data.providers.csv:CSVProvider",
}


def _target(name: str) -> str:
    if name in BUILTINS:
        return BUILTINS[name]
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return ep.value
    if ":" in name:
        return name
    raise KeyError(f"unknown data provider {name!r}; available: {sorted(available())}")


def get_provider(name: str, **options) -> DataProvider:
    module, _, attr = _target(name).partition(":")
    cls = getattr(importlib.import_module(module), attr)
    if not (isinstance(cls, type) and issubclass(cls, DataProvider)):
        raise TypeError(f"{name!r} does not resolve to a DataProvider subclass")
    return cls(**options)


def available() -> dict[str, str]:
    return {**BUILTINS, **{ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}}


__all__ = ["DataProvider", "get_provider", "available", "ENTRY_POINT_GROUP"]

"""Resolve and load strategy plugins.

A strategy reference (``strategy.ref`` in a job) can be:

* a built-in name: ``pairs``, ``momentum``, ``ma_crossover``
* an installed plugin: any name registered under the ``quantlab.strategies``
  entry-point group (``pip install my-strategies``)
* a module path: ``my_pkg.signals:MyStrategy``
* a file: ``strategies/my_strategy.py`` or ``strategies/my_strategy.py:MyStrategy``

File plugins ship their *source* inside each task payload, so they run on any
worker (process pool, Ray, Kubernetes) without being installed there. Every
reference carries a code hash. That hash is part of the run ID, so editing a
strategy never reuses stale results.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
import types
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from quantlab.strategy.base import Strategy

ENTRY_POINT_GROUP = "quantlab.strategies"
BUILTINS = {
    "pairs": "quantlab.strategies.pairs:PairsStrategy",
    "momentum": "quantlab.strategies.momentum:CrossSectionalMomentum",
    "ma_crossover": "quantlab.strategies.trend:MovingAverageCrossover",
}


@dataclass(frozen=True)
class StrategyRef:
    ref: str  # "module:Class" or "<file>.py:Class"
    code_hash: str
    source: str | None = None  # file plugins only

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def available() -> dict[str, str]:
    return {**BUILTINS, **{ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}}


def resolve(ref: str) -> StrategyRef:
    path, _, cls = ref.partition(":") if ".py" in ref else (ref, "", "")
    if path.endswith(".py"):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"strategy file not found: {p.resolve()}")
        src = p.read_text()
        return StrategyRef(f"{p.name}:{cls}" if cls else p.name, _sha(src), src)
    target = available().get(ref, ref)
    if ":" not in target:
        raise KeyError(f"unknown strategy {ref!r}; built-in/installed: {sorted(available())}, "
                       "or use 'module:Class' / 'path/to/file.py[:Class]'")
    module = importlib.import_module(target.split(":")[0])
    return StrategyRef(target, _sha(inspect.getsource(module)))


def _pick(module: types.ModuleType, cls_name: str, ref: str) -> type[Strategy]:
    if cls_name:
        cls = getattr(module, cls_name, None)
    else:
        found = [v for v in vars(module).values()
                 if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy
                 and v.__module__ == module.__name__]
        if len(found) != 1:
            raise TypeError(f"{ref}: expected exactly one Strategy subclass, found {[c.__name__ for c in found]}; "
                            "use 'file.py:ClassName'")
        cls = found[0]
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError(f"{ref} is not a quantlab Strategy subclass")
    return cls


@lru_cache(maxsize=32)
def _load(ref: str, code_hash: str, source: str | None) -> type[Strategy]:
    mod_part, _, cls_name = ref.partition(":")
    if source is not None:
        name = f"quantlab_plugin_{code_hash}"
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__file__ = mod_part
            sys.modules[name] = module
            exec(compile(source, mod_part, "exec"), module.__dict__)  # noqa: S102 - user plugin code
        return _pick(module, cls_name, ref)
    return _pick(importlib.import_module(mod_part), cls_name, ref)


def load_class(sref: StrategyRef) -> type[Strategy]:
    return _load(sref.ref, sref.code_hash, sref.source)


@lru_cache(maxsize=32)
def _instantiate(ref: str, code_hash: str, source: str | None, options: str, state: str) -> Strategy:
    return _load(ref, code_hash, source)(json.loads(options), json.loads(state))


def instantiate(sref: StrategyRef, options: dict | None = None, state: dict | None = None) -> Strategy:
    """Build (and cache per process) a strategy instance."""
    return _instantiate(sref.ref, sref.code_hash, sref.source,
                        json.dumps(options or {}, sort_keys=True), json.dumps(state or {}, sort_keys=True))

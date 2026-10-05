"""The strategy contract.

The platform owns data loading, parameter sweeps, execution, costs, validation,
the registry and recovery. A strategy owns only one idea: given market data and
a parameter set, what portfolio weights should we target at each bar?

Minimal plugin::

    class MyStrategy(Strategy):
        name = "my_strategy"

        def positions(self, data, params):
            mom = data.close.pct_change(params["lookback"])
            return np.sign(mom) / data.close.shape[1]    # bars × symbols target weights

Rules the platform enforces (``quantlab strategy check``):

* weights at bar ``t`` may only use data up to and including ``t``
  (no ``shift(-k)``, no full-sample statistics); the audit truncates history
  and fails the run if earlier weights change
* output shape is ``(len(data), len(data.symbols))``; NaN means 0
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from quantlab.data.store import MarketData

Params = dict[str, Any]


def as_weights(w: pd.DataFrame | np.ndarray, data: MarketData) -> np.ndarray:
    """Normalize a strategy's output to a float (T, N) array aligned to ``data``."""
    if isinstance(w, pd.DataFrame):
        w = w.reindex(index=data.index, columns=data.symbols)
    arr = np.asarray(w, dtype=np.float64)
    if arr.shape != (len(data), len(data.symbols)):
        raise ValueError(f"positions() returned shape {arr.shape}, expected {(len(data), len(data.symbols))}")
    return np.nan_to_num(arr, nan=0.0)


class Strategy:
    #: registry / display name
    name: ClassVar[str] = ""
    #: params that share expensive precomputation; one sweep task = one value of these
    group_by: ClassVar[tuple[str, ...]] = ()
    #: default options, overridden by ``strategy.options`` in the job spec
    default_options: ClassVar[dict[str, Any]] = {}

    def __init__(self, options: dict[str, Any] | None = None, state: dict[str, Any] | None = None):
        self.options = {**self.default_options, **(options or {})}
        self.state: dict[str, Any] = dict(state or {})

    # ------------------------------------------------------------ lifecycle hooks
    def fit(self, data: MarketData) -> dict[str, Any]:
        """Learn anything that is fixed for the whole sweep (e.g. the tradable universe or
        which pairs are cointegrated). Receives only the leading ``strategy.fit_fraction``
        of history. Must return JSON-serializable state; ``state["symbols"]`` (optional)
        restricts the symbols loaded for the backtest."""
        return {}

    def valid(self, params: Params) -> bool:
        """Filter out meaningless grid points (e.g. fast >= slow)."""
        return True

    def warmup(self, params: Params) -> int:
        """Bars needed before signals are meaningful; excluded from validation."""
        ints = [v for v in params.values() if isinstance(v, (int, np.integer)) and not isinstance(v, bool)]
        return int(max(ints, default=0))

    # ------------------------------------------------------------ the strategy
    def positions(self, data: MarketData, params: Params) -> pd.DataFrame | np.ndarray:
        """Target weights, bars × symbols. Override this (simple) or ``positions_batch`` (fast)."""
        raise NotImplementedError

    def positions_batch(self, data: MarketData, params_list: list[Params]) -> np.ndarray:
        """Target weights for many param sets at once → (T, N, C). Every entry of
        ``params_list`` shares the same ``group_by`` values, so vectorized overrides can
        compute shared state once."""
        return np.stack([as_weights(self.positions(data, p), data) for p in params_list], axis=-1)

    # ------------------------------------------------------------ validation / production
    def checks(self, data: MarketData, params: Params, windows: list[tuple[int, int]]) -> list[dict[str, Any]]:
        """Extra strategy-specific validation gates: ``{"check", "value", "rule", "passed"}``.
        ``windows`` are the walk-forward train windows as bar index ranges."""
        return []

    def describe(self, data: MarketData, params: Params) -> dict[str, Any]:
        """Human-readable snapshot of the strategy's current state for the registry."""
        return {}

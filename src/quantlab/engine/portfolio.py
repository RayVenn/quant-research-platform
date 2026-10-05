"""Vectorized portfolio simulator shared by every strategy.

Strategies emit *target weights* ``W[t, n, c]``: the fraction of capital in asset
``n`` decided from data up to and including bar ``t``, for parameter combo ``c``.
The simulator applies one convention to all of them:

* execution: a target set at bar ``t`` is filled at the close of ``t + delay``,
  so P&L accrues from bar ``t + delay + 1`` (``held = W`` lagged ``1 + delay``)
* P&L: ``Σ_n held[t, n] · r[t, n]`` with close-to-close simple returns
* costs: ``cost_bps`` per unit of traded notional ``Σ_n |held_t − held_{t−1}|``
* borrow: ``borrow_bps`` per year on short notional
* weights are constant-weight rebalanced each bar (the standard vectorized
  research approximation); untradable assets (NaN price) have zero weight

Everything is numpy broadcasting over a ``(T, N, C)`` cube. A whole parameter
chunk is simulated in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    returns: np.ndarray  # (T, C) net portfolio returns
    traded: np.ndarray  # (T, C) traded notional
    gross: np.ndarray  # (T, C) gross exposure held
    net: np.ndarray  # (T, C) net exposure held


def lag(a: np.ndarray, k: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill)
    if k < a.shape[0]:
        out[k:] = a[: a.shape[0] - k]
    return out


def simulate(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    cost_bps: float = 1.0,
    delay: int = 1,
    borrow_bps: float = 0.0,
    periods_per_year: float = 252.0,
) -> SimResult:
    if weights.ndim == 2:
        weights = weights[:, :, None]
    if weights.shape[:2] != asset_returns.shape:
        raise ValueError(f"weights {weights.shape[:2]} and returns {asset_returns.shape} disagree on (T, N)")
    w = np.nan_to_num(weights.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    held = lag(w, 1 + delay)
    prev = lag(held, 1)
    r = np.nan_to_num(asset_returns)[:, :, None]
    traded = np.abs(held - prev).sum(axis=1)
    ret = (held * r).sum(axis=1) - cost_bps / 1e4 * traded
    if borrow_bps:
        ret -= borrow_bps / 1e4 / periods_per_year * np.clip(-held, 0, None).sum(axis=1)
    return SimResult(ret, traded, np.abs(held).sum(axis=1), held.sum(axis=1))

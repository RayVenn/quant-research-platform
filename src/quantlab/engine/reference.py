"""Bar-by-bar reference simulator — the correctness oracle for ``engine.portfolio``."""

from __future__ import annotations

import numpy as np


def simulate_loop(
    weights: np.ndarray,
    prices: np.ndarray,
    cost_bps: float = 1.0,
    delay: int = 1,
    borrow_bps: float = 0.0,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Net returns (T,) for target weights (T, N) on prices (T, N), using explicit loops."""
    T, N = weights.shape
    lag = 1 + delay
    ret = np.zeros(T)
    prev = np.zeros(N)
    for t in range(T):
        held = np.zeros(N)
        if t >= lag:
            for n in range(N):
                w = weights[t - lag, n]
                held[n] = 0.0 if not np.isfinite(w) else w
        pnl = 0.0
        if t > 0:
            for n in range(N):
                if np.isfinite(prices[t, n]) and np.isfinite(prices[t - 1, n]):
                    pnl += held[n] * (prices[t, n] / prices[t - 1, n] - 1.0)
        cost = cost_bps / 1e4 * np.abs(held - prev).sum()
        borrow = borrow_bps / 1e4 / periods_per_year * sum(max(-h, 0.0) for h in held)
        ret[t] = pnl - cost - borrow
        prev = held
    return ret

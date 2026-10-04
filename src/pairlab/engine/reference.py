"""Bar-by-bar reference implementation of the engine, used to verify the vectorized path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairlab.engine.vectorized import hedge_and_zscore


def run_pair_loop(
    price_y: pd.Series,
    price_x: pd.Series,
    beta_window: int,
    z_window: int,
    entry: float,
    exit_: float,
    stop: float,
    cost_bps: float = 1.0,
    delay: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (returns, held) arrays for one pair and one parameter set, using explicit loops."""
    ly, lx = np.log(price_y.to_frame()), np.log(price_x.to_frame())
    beta, z = hedge_and_zscore(ly, lx, beta_window, z_window)
    beta, z = beta[:, 0], z[:, 0]
    py, px = price_y.to_numpy(), price_x.to_numpy()
    n = len(z)

    pos = np.zeros(n, dtype=np.int8)
    state = 0
    for t in range(n):
        zt = z[t]
        if np.isnan(zt) or abs(zt) < exit_ or abs(zt) >= stop:
            state = 0
        elif zt > entry:
            state = -1
        elif zt < -entry:
            state = 1
        pos[t] = state

    lag = 1 + delay
    ret = np.zeros(n)
    held = np.zeros(n, dtype=np.int8)
    for t in range(n):
        held[t] = pos[t - lag] if t >= lag else 0
        if t == 0:
            continue
        b = beta[t - lag] if t >= lag else np.nan
        pnl = 0.0
        if held[t] != 0 and not np.isnan(b):
            ry = py[t] / py[t - 1] - 1.0
            rx = px[t] / px[t - 1] - 1.0
            pnl = held[t] * (ry - b * rx) / (1.0 + abs(b))
        ret[t] = pnl - cost_bps / 1e4 * abs(int(held[t]) - int(held[t - 1]))
    return ret, held

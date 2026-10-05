"""Synthetic market generator with known cointegrated structure.

Used for tests, demos and as a smoke dataset in fresh environments, so a
researcher can run the full pipeline minutes after provisioning without any
vendor credentials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_universe(
    n_clusters: int = 4,
    per_cluster: int = 3,
    n_noise: int = 4,
    n_bars: int = 2520,
    freq: str = "B",
    start: str = "2016-01-01",
    half_life: float = 8.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a wide close-price panel.

    Each cluster shares a random-walk factor; members load on it with a random
    beta plus mean-reverting (OU) idiosyncratic noise, so any two members of a
    cluster are cointegrated. ``n_noise`` independent random walks act as
    decoys the pair screen should reject.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    phi = np.exp(-np.log(2) / half_life)
    cols: dict[str, np.ndarray] = {}

    for c in range(n_clusters):
        factor = np.cumsum(rng.normal(0.0003, 0.012, n_bars)) + np.log(rng.uniform(20, 200))
        for m in range(per_cluster):
            beta = rng.uniform(0.6, 1.4)
            eps = rng.normal(0, 0.012, n_bars)
            ou = np.zeros(n_bars)
            for t in range(1, n_bars):
                ou[t] = phi * ou[t - 1] + eps[t]
            log_p = beta * factor + rng.uniform(-0.5, 0.5) + ou
            cols[f"C{c}M{m}"] = np.exp(log_p)

    for k in range(n_noise):
        log_p = np.cumsum(rng.normal(0.0002, 0.015, n_bars)) + np.log(rng.uniform(20, 200))
        cols[f"N{k}"] = np.exp(log_p)

    return pd.DataFrame(cols, index=index)

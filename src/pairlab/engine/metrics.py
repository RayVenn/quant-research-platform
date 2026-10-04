"""Performance metrics vectorized along the time axis (axis 0).

Every function accepts arrays shaped ``(T, ...)`` and returns arrays shaped
``(...)`` so one call scores thousands of (pair, parameter) series at once.
"""

from __future__ import annotations

import numpy as np

METRIC_COLUMNS = (
    "sharpe",
    "ann_return",
    "ann_vol",
    "total_return",
    "max_drawdown",
    "n_trades",
    "exposure",
    "turnover",
)


def sharpe(ret: np.ndarray, periods_per_year: float) -> np.ndarray:
    mu = ret.mean(axis=0)
    sd = ret.std(axis=0, ddof=1) if ret.shape[0] > 1 else np.zeros_like(mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 1e-12, mu / sd * np.sqrt(periods_per_year), 0.0)
    return out


def max_drawdown(ret: np.ndarray) -> np.ndarray:
    equity = np.exp(np.cumsum(np.log1p(ret), axis=0))
    peak = np.maximum.accumulate(np.maximum(equity, 1.0), axis=0)
    return (equity / peak - 1.0).min(axis=0)


def score(
    ret: np.ndarray,
    held: np.ndarray | None,
    periods_per_year: float,
) -> dict[str, np.ndarray]:
    """Compute the standard metric set for return series ``ret`` (T, ...).

    ``held`` is the position actually held each bar (same shape); it drives the
    activity metrics. Pass ``None`` for portfolio-level series.
    """
    n = ret.shape[0]
    log_growth = np.log1p(ret).sum(axis=0)
    total = np.expm1(log_growth)
    years = n / periods_per_year
    out = {
        "sharpe": sharpe(ret, periods_per_year),
        "ann_return": np.expm1(log_growth / years),
        "ann_vol": ret.std(axis=0, ddof=1) * np.sqrt(periods_per_year),
        "total_return": total,
        "max_drawdown": max_drawdown(ret),
    }
    if held is not None:
        prev = np.concatenate([np.zeros_like(held[:1]), held[:-1]], axis=0)
        entries = (held != 0) & (held != prev)
        out["n_trades"] = entries.sum(axis=0).astype(np.int64)
        out["exposure"] = (held != 0).mean(axis=0)
        out["turnover"] = np.abs(held.astype(np.float64) - prev).sum(axis=0) / years
    return out

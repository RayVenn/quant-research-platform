"""Performance metrics vectorized along the time axis (axis 0).

Every function accepts arrays shaped ``(T, ...)`` and returns arrays shaped
``(...)``, so one call scores thousands of parameter combos at once.
"""

from __future__ import annotations

import numpy as np

METRIC_COLUMNS = (
    "sharpe",
    "ann_return",
    "ann_vol",
    "total_return",
    "max_drawdown",
    "turnover",
    "avg_gross",
    "pct_invested",
)


def sharpe(ret: np.ndarray, periods_per_year: float) -> np.ndarray:
    mu = ret.mean(axis=0)
    sd = ret.std(axis=0, ddof=1) if ret.shape[0] > 1 else np.zeros_like(mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 1e-12, mu / sd * np.sqrt(periods_per_year), 0.0)


def max_drawdown(ret: np.ndarray) -> np.ndarray:
    equity = np.exp(np.cumsum(np.log1p(ret), axis=0))
    peak = np.maximum.accumulate(np.maximum(equity, 1.0), axis=0)
    return (equity / peak - 1.0).min(axis=0)


def score(
    ret: np.ndarray,
    periods_per_year: float,
    traded: np.ndarray | None = None,
    gross: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Standard metric set for return series ``ret`` (T, ...)."""
    n = ret.shape[0]
    log_growth = np.log1p(ret).sum(axis=0)
    years = n / periods_per_year
    out = {
        "sharpe": sharpe(ret, periods_per_year),
        "ann_return": np.expm1(log_growth / years),
        "ann_vol": ret.std(axis=0, ddof=1) * np.sqrt(periods_per_year),
        "total_return": np.expm1(log_growth),
        "max_drawdown": max_drawdown(ret),
    }
    if traded is not None:
        out["turnover"] = traded.sum(axis=0) / years
    if gross is not None:
        out["avg_gross"] = gross.mean(axis=0)
        out["pct_invested"] = (gross > 1e-12).mean(axis=0)
    return out

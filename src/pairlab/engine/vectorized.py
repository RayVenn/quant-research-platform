"""Vectorized pairs-trading backtest engine.

Model
-----
For a pair (Y, X) with log prices ``ly``, ``lx``:

* hedge ratio ``beta_t``  = rolling OLS slope of ly on lx over ``beta_window`` bars
* spread ``s_t``          = rolling OLS residual: (ly_t - mean_w(ly)) - beta_t * (lx_t - mean_w(lx))
* z-score ``z_t``         = (s_t - mean_w(s)) / std_w(s) over ``z_window`` bars
* target position (hysteresis, per threshold combo):
    |z| >= stop           -> 0   (stop-loss, also blocks entries)
    |z| <  exit           -> 0   (mean reversion achieved)
    entry < z  < stop     -> -1  (short spread: short Y, long X)
    -stop < z < -entry    -> +1  (long spread:  long Y, short X)
    otherwise             -> hold previous position
* execution: a signal on bar ``t`` is filled at the close of ``t + delay``;
  P&L accrues from the following bar. All windows are trailing, so there is
  no lookahead.
* P&L: dollar weights ``1/(1+|b|)`` on Y and ``-b/(1+|b|)`` on X (gross = 1),
  using the hedge ratio at signal time. Costs are ``cost_bps`` per unit of
  gross notional traded.

Vectorization
-------------
Everything except the (cheap) rolling statistics is pure numpy broadcasting
over a ``(T, P, C)`` cube — T bars, P pairs, C threshold combos. The
hysteresis state machine is expressed as "events + forward fill", which turns
a sequential loop into an index ``maximum.accumulate``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairlab.engine import metrics


@dataclass(frozen=True)
class Thresholds:
    entry: np.ndarray
    exit: np.ndarray
    stop: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.entry) == len(self.exit) == len(self.stop)):
            raise ValueError("threshold arrays must have equal length")

    def __len__(self) -> int:
        return len(self.entry)

    def slice(self, sl: slice) -> Thresholds:
        return Thresholds(self.entry[sl], self.exit[sl], self.stop[sl])


@dataclass
class GroupResult:
    """Output of one (beta_window, z_window) group across P pairs and C combos."""

    pair_metrics: dict[str, np.ndarray]  # each (P, C)
    portfolio_returns: np.ndarray  # (T, C) equal-weight across pairs
    index: pd.DatetimeIndex


def hedge_and_zscore(
    ly: pd.DataFrame, lx: pd.DataFrame, beta_window: int, z_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling hedge ratio and spread z-score, each (T, P). Columns of ly/lx are aligned by position."""
    cols = range(ly.shape[1])
    y = pd.DataFrame(ly.to_numpy(), index=ly.index, columns=cols)
    x = pd.DataFrame(lx.to_numpy(), index=lx.index, columns=cols)
    ry, rx = y.rolling(beta_window), x.rolling(beta_window)
    beta = ry.cov(x) / rx.var()
    spread = (y - ry.mean()) - beta * (x - rx.mean())
    rs = spread.rolling(z_window)
    sd = rs.std()
    z = (spread - rs.mean()) / sd.where(sd > 1e-12)
    return beta.to_numpy(), z.to_numpy()


def target_positions(z: np.ndarray, th: Thresholds) -> np.ndarray:
    """Hysteresis positions (T, P, C) as int8 via events + forward fill."""
    zz = z[:, :, None]
    az = np.abs(zz)
    entry, exit_, stop = (a[None, None, :] for a in (th.entry, th.exit, th.stop))
    with np.errstate(invalid="ignore"):
        flat = np.isnan(zz) | (az < exit_) | (az >= stop)
        short = zz > entry
        long_ = zz < -entry
    event = np.where(flat, 0, np.where(short, -1, np.where(long_, 1, 127))).astype(np.int8)
    has_event = event != 127
    t_idx = np.arange(z.shape[0], dtype=np.int32)[:, None, None]
    last = np.maximum.accumulate(np.where(has_event, t_idx, 0), axis=0)
    pos = np.take_along_axis(event, last, axis=0)
    pos[pos == 127] = 0
    return pos


def _lag(a: np.ndarray, k: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(a, fill)
    if k < a.shape[0]:
        out[k:] = a[: a.shape[0] - k]
    return out


def run_group(
    prices_y: pd.DataFrame,
    prices_x: pd.DataFrame,
    beta_window: int,
    z_window: int,
    thresholds: Thresholds,
    cost_bps: float = 1.0,
    delay: int = 1,
    periods_per_year: float = 252.0,
    max_cells: int = 20_000_000,
) -> GroupResult:
    """Backtest every pair (columns) × every threshold combo for one window group.

    ``max_cells`` bounds peak memory: combos are processed in chunks so that
    T × P × chunk stays under the limit.
    """
    if prices_y.shape != prices_x.shape:
        raise ValueError("prices_y and prices_x must have the same shape")
    ly, lx = np.log(prices_y), np.log(prices_x)
    beta, z = hedge_and_zscore(ly, lx, beta_window, z_window)
    T, P = z.shape

    py, px = prices_y.to_numpy(), prices_x.to_numpy()
    ret_y = np.nan_to_num(_lag(py, 0) / _lag(py, 1, np.nan) - 1.0)
    ret_x = np.nan_to_num(_lag(px, 0) / _lag(px, 1, np.nan) - 1.0)
    hb = _lag(beta, 1 + delay, np.nan)
    with np.errstate(invalid="ignore"):
        spread_ret = np.nan_to_num((ret_y - hb * ret_x) / (1.0 + np.abs(hb)))
    cost = cost_bps / 1e4

    C = len(thresholds)
    chunk = max(1, min(C, max_cells // max(1, T * P)))
    pair_metrics: dict[str, list[np.ndarray]] = {k: [] for k in metrics.METRIC_COLUMNS}
    port = np.empty((T, C))
    for start in range(0, C, chunk):
        th = thresholds.slice(slice(start, start + chunk))
        pos = target_positions(z, th)
        held = _lag(pos, 1 + delay, 0)
        prev = _lag(held, 1, 0)
        ret = held * spread_ret[:, :, None] - cost * np.abs(held - prev)
        scored = metrics.score(ret, held, periods_per_year)
        for k in metrics.METRIC_COLUMNS:
            pair_metrics[k].append(scored[k])
        port[:, start : start + len(th)] = ret.mean(axis=1)

    return GroupResult(
        pair_metrics={k: np.concatenate(v, axis=1) for k, v in pair_metrics.items()},
        portfolio_returns=port,
        index=prices_y.index,
    )

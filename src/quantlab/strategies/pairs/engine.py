"""Vectorized pairs-trading signal engine (hedge ratios, z-scores, hysteresis positions).

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
* weights: ``pos/(1+|b|)`` on Y and ``-pos·b/(1+|b|)`` on X (gross = 1 per pair),
  equal-weighted across pairs. Execution, costs and P&L are applied by the
  shared portfolio simulator. All windows are trailing, so there is no lookahead.

Vectorization
-------------
Everything except the (cheap) rolling statistics is pure numpy broadcasting
over a ``(T, P, C)`` cube (T bars, P pairs, C threshold combos). The
hysteresis state machine is expressed as "events + forward fill", which turns
a sequential loop into an index ``maximum.accumulate``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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

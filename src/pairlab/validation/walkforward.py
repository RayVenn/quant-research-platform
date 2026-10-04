"""Walk-forward analysis on precomputed per-configuration return series.

Because the engine only uses trailing windows, the return of configuration
``c`` at bar ``t`` never depends on data after ``t``. That lets us run the
sweep once over the full history and then *slice* it into train/test folds,
instead of re-running the backtest per fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pairlab.engine import metrics as M


@dataclass(frozen=True)
class Fold:
    k: int
    train_start: int
    train_end: int  # exclusive; == test_start
    test_end: int  # exclusive


def make_folds(n_bars: int, train_bars: int, test_bars: int, scheme: str = "rolling", warmup: int = 0) -> list[Fold]:
    folds, k = [], 0
    test_start = warmup + train_bars
    while test_start + test_bars <= n_bars:
        train_start = warmup if scheme == "expanding" else test_start - train_bars
        folds.append(Fold(k, train_start, test_start, test_start + test_bars))
        test_start += test_bars
        k += 1
    if not folds:
        raise ValueError(
            f"not enough history for a single fold: n_bars={n_bars}, warmup={warmup}, "
            f"train={train_bars}, test={test_bars}"
        )
    return folds


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame  # one row per fold
    oos_returns: pd.Series  # stitched out-of-sample returns of the selected config


def walk_forward(returns: pd.DataFrame, folds: list[Fold], periods_per_year: float) -> WalkForwardResult:
    arr = returns.to_numpy()
    cols = returns.columns
    rows, pieces = [], []
    for f in folds:
        train, test = arr[f.train_start : f.train_end], arr[f.train_end : f.test_end]
        is_sr = M.sharpe(train, periods_per_year)
        best = int(np.argmax(is_sr))
        oos = test[:, best]
        rows.append(
            {
                "fold": f.k,
                "train_start": returns.index[f.train_start],
                "test_start": returns.index[f.train_end],
                "test_end": returns.index[f.test_end - 1],
                "selected": cols[best],
                "is_sharpe": float(is_sr[best]),
                "oos_sharpe": float(M.sharpe(oos[:, None], periods_per_year)[0]),
                "oos_return": float(np.expm1(np.log1p(oos).sum())),
            }
        )
        pieces.append(pd.Series(oos, index=returns.index[f.train_end : f.test_end]))
    return WalkForwardResult(pd.DataFrame(rows), pd.concat(pieces))

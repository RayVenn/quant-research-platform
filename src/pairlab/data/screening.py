"""Candidate pair screening: vectorized correlation pre-filter + Engle-Granger test."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


@dataclass(frozen=True)
class PairCandidate:
    y: str
    x: str
    corr: float
    pvalue: float
    hedge_ratio: float
    half_life: float

    def to_dict(self) -> dict:
        return asdict(self)


def half_life(spread: np.ndarray) -> float:
    """OU half-life (bars) from an AR(1) fit of Δs on s_{t-1}."""
    s = np.asarray(spread, dtype=float)
    lag, delta = s[:-1], np.diff(s)
    lag_c = lag - lag.mean()
    denom = float(lag_c @ lag_c)
    if denom == 0:
        return float("inf")
    b = float(lag_c @ (delta - delta.mean())) / denom
    return float(-np.log(2) / b) if b < 0 else float("inf")


def engle_granger(log_y: np.ndarray, log_x: np.ndarray) -> tuple[float, float, float]:
    """Return (p-value, hedge ratio, half-life) for log_y ~ log_x."""
    _, pvalue, _ = coint(log_y, log_x)
    beta, alpha = np.polyfit(log_x, log_y, 1)
    hl = half_life(log_y - beta * log_x - alpha)
    return float(pvalue), float(beta), hl


def screen_pairs(
    panel: pd.DataFrame,
    min_corr: float = 0.7,
    max_pvalue: float = 0.05,
    max_half_life: float = 60.0,
    max_pairs: int = 50,
) -> list[PairCandidate]:
    """Screen all symbol combinations in *panel* for tradeable cointegration.

    Correlation of log returns is computed for every pair at once (one matrix
    op); only pairs above ``min_corr`` pay for the more expensive EG test.
    """
    clean = panel.dropna(axis=1, how="any")
    logp = np.log(clean)
    corr = logp.diff().dropna().corr().to_numpy()
    syms = list(clean.columns)
    out: list[PairCandidate] = []
    for i, j in itertools.combinations(range(len(syms)), 2):
        if corr[i, j] < min_corr:
            continue
        p, beta, hl = engle_granger(logp.iloc[:, i].to_numpy(), logp.iloc[:, j].to_numpy())
        if p <= max_pvalue and hl <= max_half_life and beta > 0:
            out.append(PairCandidate(syms[i], syms[j], float(corr[i, j]), p, beta, hl))
    out.sort(key=lambda c: c.pvalue)
    return out[:max_pairs]

"""Statistics for judging whether a backtest result is real.

References
----------
* Bailey & López de Prado (2014), "The Deflated Sharpe Ratio".
* Bailey, Borwein, López de Prado & Zhu (2015), "The Probability of Backtest Overfitting".
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def per_period_sharpe(ret: np.ndarray) -> np.ndarray:
    sd = ret.std(axis=0, ddof=1)
    return np.where(sd > 1e-12, ret.mean(axis=0) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def probabilistic_sharpe(ret: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) given a single return series (per-period units)."""
    ret = np.asarray(ret, dtype=float)
    n = len(ret)
    if n < 3 or ret.std() == 0:
        return 0.0
    sr = float(per_period_sharpe(ret))
    skew = float(stats.skew(ret))
    kurt = float(stats.kurtosis(ret, fisher=False))
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def effective_trials(trial_returns: np.ndarray) -> float:
    """Effective number of independent trials, N_eff = rho + (1 - rho) * N, from mean pairwise correlation.

    Neighbouring grid points are highly correlated, so counting every config as an
    independent trial would over-deflate.
    """
    n = trial_returns.shape[1]
    if n < 2:
        return float(n)
    active = trial_returns[:, trial_returns.std(axis=0) > 0]
    if active.shape[1] < 2:
        return 1.0
    c = np.corrcoef(active, rowvar=False)
    rho = float((c.sum() - np.trace(c)) / (c.shape[0] * (c.shape[0] - 1)))
    rho = min(max(rho, 0.0), 1.0)
    return max(1.0, rho + (1 - rho) * n)


def expected_max_sharpe(trial_sharpes: np.ndarray, n_trials: float | None = None) -> float:
    """Expected maximum per-period Sharpe among N unskilled trials with the observed dispersion."""
    n = float(n_trials if n_trials is not None else len(trial_sharpes))
    if n < 2 or len(trial_sharpes) < 2:
        return 0.0
    sd = float(np.std(trial_sharpes, ddof=1))
    return sd * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n) + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n * math.e))
    )


def deflated_sharpe(ret: np.ndarray, trial_sharpes: np.ndarray, n_trials: float | None = None) -> float:
    """PSR against the Sharpe you would expect from the best of N skill-less trials."""
    return probabilistic_sharpe(ret, expected_max_sharpe(np.asarray(trial_sharpes, dtype=float), n_trials))


def pbo_cscv(returns: np.ndarray, n_splits: int = 10, max_combinations: int = 2000, seed: int = 0) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

    ``returns`` is (T, N): one column per strategy configuration tried. For every
    way of choosing half the time blocks as in-sample, pick the IS-best config
    and see where it ranks out-of-sample. PBO = P(IS-best is below the OOS median).
    """
    T, N = returns.shape
    if N < 2:
        return {"pbo": float("nan"), "n_combinations": 0, "logits": []}
    n_splits -= n_splits % 2
    blocks = np.array_split(np.arange(T), n_splits)
    s1 = np.stack([returns[b].sum(axis=0) for b in blocks])
    s2 = np.stack([(returns[b] ** 2).sum(axis=0) for b in blocks])
    cnt = np.array([len(b) for b in blocks], dtype=float)

    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]

    def _sr(idx: np.ndarray) -> np.ndarray:
        n = cnt[idx].sum()
        mu = s1[idx].sum(axis=0) / n
        var = np.maximum(s2[idx].sum(axis=0) / n - mu**2, 0)
        return np.where(var > 1e-18, mu / np.sqrt(np.where(var > 0, var, 1)), 0.0)

    logits, degradation = [], []
    all_idx = np.arange(n_splits)
    for is_idx in combos:
        is_idx = np.array(is_idx)
        oos_idx = np.setdiff1d(all_idx, is_idx)
        sr_is, sr_oos = _sr(is_idx), _sr(oos_idx)
        best = int(np.argmax(sr_is))
        rank = stats.rankdata(sr_oos)[best]
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
        degradation.append((float(sr_is[best]), float(sr_oos[best])))
    logits_arr = np.array(logits)
    return {
        "pbo": float((logits_arr <= 0).mean()),
        "n_combinations": len(combos),
        "logits": logits_arr.tolist(),
        "is_vs_oos_sharpe": degradation,
    }

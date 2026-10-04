"""Model validation for any strategy: walk-forward OOS, deflated Sharpe, PBO, plus the
strategy's own checks, all evaluated against promotion gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.store import MarketData
from quantlab.engine import metrics as M
from quantlab.strategy.base import Params, Strategy
from quantlab.sweep.spec import JobSpec
from quantlab.validation.stats import deflated_sharpe, effective_trials, pbo_cscv, per_period_sharpe
from quantlab.validation.walkforward import make_folds, walk_forward


def _py(v: Any) -> Any:
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def validate(
    spec: JobSpec, market: MarketData, strategy: Strategy, combos: dict[str, Params], returns: pd.DataFrame
) -> dict:
    v, g, ppy = spec.validation, spec.gates, spec.data.periods_per_year
    warmup = v.warmup_bars if v.warmup_bars is not None else max(strategy.warmup(p) for p in combos.values())
    returns = returns.reindex(market.index).fillna(0.0)
    folds = make_folds(len(returns), v.train_bars, v.test_bars, v.scheme, warmup)
    research = returns.iloc[warmup:]

    wf = walk_forward(returns, folds, ppy)
    oos = wf.oos_returns.to_numpy()
    oos_scores = {k: float(val[0]) for k, val in M.score(oos[:, None], ppy).items()}

    trial_sr = per_period_sharpe(research.to_numpy())
    n_eff = effective_trials(research.to_numpy())
    dsr = deflated_sharpe(oos, trial_sr, n_eff)
    pbo = pbo_cscv(research.to_numpy(), v.pbo_splits)

    chosen = research.columns[int(np.argmax(M.sharpe(research.to_numpy(), ppy)))]
    params = combos[chosen]

    checks: dict[str, tuple[Any, bool, str]] = {
        "oos_sharpe": (oos_scores["sharpe"], oos_scores["sharpe"] >= g.min_oos_sharpe, f">= {g.min_oos_sharpe}"),
        "deflated_sharpe": (dsr, dsr >= g.min_dsr, f">= {g.min_dsr}"),
        "pbo": (pbo["pbo"], pbo["pbo"] <= g.max_pbo, f"<= {g.max_pbo}"),
        "oos_max_drawdown": (
            oos_scores["max_drawdown"], oos_scores["max_drawdown"] >= g.max_oos_drawdown, f">= {g.max_oos_drawdown}"
        ),
    }
    details = {}
    for c in strategy.checks(market, params, [(f.train_start, f.train_end) for f in folds]):
        checks[c["check"]] = (c["value"], bool(c["passed"]), c["rule"])
        if "detail" in c:
            details[c["check"]] = c["detail"]

    return {
        "passed": all(ok for _, ok, _ in checks.values()),
        "checks": {k: {"value": _py(val), "passed": bool(ok), "rule": rule} for k, (val, ok, rule) in checks.items()},
        "combo_id": chosen,
        "params": params,
        "n_trials": int(research.shape[1]),
        "n_effective_trials": round(float(n_eff), 2),
        "warmup_bars": int(warmup),
        "oos": oos_scores,
        "walk_forward": wf.folds.assign(
            train_start=lambda d: d.train_start.astype(str),
            test_start=lambda d: d.test_start.astype(str),
            test_end=lambda d: d.test_end.astype(str),
        ).to_dict(orient="records"),
        "pbo_detail": {"n_combinations": pbo["n_combinations"]},
        "check_details": details,
        "snapshot": strategy.describe(market, params),
        "oos_equity": wf.oos_returns.pipe(lambda s: (1 + s).cumprod()),
    }

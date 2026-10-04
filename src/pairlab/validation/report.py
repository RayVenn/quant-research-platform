"""Model validation: walk-forward OOS, deflated Sharpe, PBO, cointegration stability, promotion gates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairlab.data.screening import engle_granger
from pairlab.engine import metrics as M
from pairlab.engine.vectorized import hedge_and_zscore
from pairlab.sweep.spec import JobSpec
from pairlab.validation.stats import deflated_sharpe, effective_trials, pbo_cscv, per_period_sharpe
from pairlab.validation.walkforward import make_folds, walk_forward


def cointegration_stability(
    panel: pd.DataFrame, pairs: list[tuple[str, str]], windows: list[tuple[int, int]], pvalue: float
) -> dict[str, float]:
    logp = np.log(panel)
    out = {}
    for y, x in pairs:
        passed = [
            engle_granger(logp[y].to_numpy()[a:b], logp[x].to_numpy()[a:b])[0] < pvalue for a, b in windows
        ]
        out[f"{y}/{x}"] = float(np.mean(passed))
    return out


def validate(
    spec: JobSpec,
    panel: pd.DataFrame,
    pairs: list[tuple[str, str]],
    pair_metrics: pd.DataFrame,
    returns: pd.DataFrame,
) -> dict:
    v, g, ppy = spec.validation, spec.gates, spec.data.periods_per_year
    warmup = v.warmup_bars if v.warmup_bars is not None else max(spec.grid.beta_window) + max(spec.grid.z_window)
    returns = returns.loc[panel.index]
    folds = make_folds(len(returns), v.train_bars, v.test_bars, v.scheme, warmup)
    research = returns.iloc[warmup:]

    wf = walk_forward(returns, folds, ppy)
    oos = wf.oos_returns.to_numpy()
    oos_scores = {k: float(val[0]) for k, val in M.score(oos[:, None], None, ppy).items()}

    trial_sr = per_period_sharpe(research.to_numpy())
    n_eff = effective_trials(research.to_numpy())
    dsr = deflated_sharpe(oos, trial_sr, n_eff)
    pbo = pbo_cscv(research.to_numpy(), v.pbo_splits)

    full_sr = M.sharpe(research.to_numpy(), ppy)
    chosen = research.columns[int(np.argmax(full_sr))]
    chosen_rows = pair_metrics[pair_metrics["combo_id"] == chosen].iloc[0]
    params = {
        "combo_id": chosen,
        "beta_window": int(chosen_rows["beta_window"]),
        "z_window": int(chosen_rows["z_window"]),
        "entry_z": float(chosen_rows["entry_z"]),
        "exit_z": float(chosen_rows["exit_z"]),
        "stop_z": float(chosen_rows["stop_z"]),
        "cost_bps": spec.costs.cost_bps,
        "delay": spec.costs.delay,
    }

    stability = cointegration_stability(
        panel, pairs, [(f.train_start, f.train_end) for f in folds], v.coint_pvalue
    )
    per_pair = pair_metrics[pair_metrics["combo_id"] == chosen].set_index("pair")
    ly = np.log(panel[[y for y, _ in pairs]])
    lx = np.log(panel[[x for _, x in pairs]])
    beta, z = hedge_and_zscore(ly, lx, params["beta_window"], params["z_window"])
    pair_rows = []
    for i, (y, x) in enumerate(pairs):
        label = f"{y}/{x}"
        pair_rows.append(
            {
                "y": y,
                "x": x,
                "coint_stability": stability[label],
                "in_sample_sharpe": float(per_pair.loc[label, "sharpe"]),
                "n_trades": int(per_pair.loc[label, "n_trades"]),
                "latest_beta": float(beta[-1, i]),
                "latest_z": float(z[-1, i]),
                "selected": stability[label] >= v.min_coint_stability,
            }
        )
    n_selected = sum(p["selected"] for p in pair_rows)

    checks = {
        "oos_sharpe": (oos_scores["sharpe"], oos_scores["sharpe"] >= g.min_oos_sharpe, f">= {g.min_oos_sharpe}"),
        "deflated_sharpe": (dsr, dsr >= g.min_dsr, f">= {g.min_dsr}"),
        "pbo": (pbo["pbo"], pbo["pbo"] <= g.max_pbo, f"<= {g.max_pbo}"),
        "oos_max_drawdown": (
            oos_scores["max_drawdown"], oos_scores["max_drawdown"] >= g.max_oos_drawdown, f">= {g.max_oos_drawdown}"
        ),
        "n_stable_pairs": (n_selected, n_selected >= g.min_pairs, f">= {g.min_pairs}"),
    }
    return {
        "passed": all(ok for _, ok, _ in checks.values()),
        "checks": {k: {"value": val, "passed": bool(ok), "rule": rule} for k, (val, ok, rule) in checks.items()},
        "params": params,
        "pairs": pair_rows,
        "n_trials": int(research.shape[1]),
        "n_effective_trials": round(n_eff, 2),
        "oos": oos_scores,
        "walk_forward": wf.folds.assign(
            train_start=lambda d: d.train_start.astype(str),
            test_start=lambda d: d.test_start.astype(str),
            test_end=lambda d: d.test_end.astype(str),
        ).to_dict(orient="records"),
        "pbo_detail": {"n_combinations": pbo["n_combinations"]},
        "oos_equity": wf.oos_returns.pipe(lambda s: (1 + s).cumprod()),
    }

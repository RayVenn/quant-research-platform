"""Task execution — runs on any worker (in-process, subprocess, or Ray node).

A task payload is fully self-describing (data location, strategy code and
state, parameters, output location), so workers are stateless and any task can run anywhere.
Outputs are written atomically; a task whose outputs exist is complete.
"""

from __future__ import annotations

import os
import random
import socket
import time
from typing import Any

import fsspec
import numpy as np
import pandas as pd

from quantlab.data.store import cached_market, tradable_mask
from quantlab.engine import metrics as M
from quantlab.engine.portfolio import simulate
from quantlab.strategy.loader import StrategyRef, instantiate

CHAOS_ENV = "QUANTLAB_CHAOS_FAIL_RATE"


def output_paths(results_dir: str, task_id: str) -> tuple[str, str]:
    return f"{results_dir}/{task_id}.metrics.parquet", f"{results_dir}/{task_id}.returns.parquet"


def outputs_exist(results_dir: str, task_id: str) -> bool:
    fs, _ = fsspec.core.url_to_fs(results_dir)
    return all(fs.exists(p) for p in output_paths(results_dir, task_id))


def _atomic_write(df: pd.DataFrame, path: str) -> None:
    fs, p = fsspec.core.url_to_fs(path)
    fs.makedirs(p.rsplit("/", 1)[0], exist_ok=True)
    tmp = f"{p}.tmp-{os.getpid()}"
    with fs.open(tmp, "wb") as fh:
        df.to_parquet(fh, index=True)
    fs.mv(tmp, p)


def _maybe_inject_failure(payload: dict[str, Any]) -> None:
    rate = float(os.environ.get(CHAOS_ENV, "0") or 0)
    if rate > 0 and random.Random(f"{payload['task_id']}:{payload.get('attempt', 1)}").random() < rate:
        raise RuntimeError(f"chaos: injected failure (rate={rate})")


def execute_task(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    _maybe_inject_failure(payload)

    d = payload["data"]
    market = cached_market(d["store"], tuple(d["symbols"]), d.get("start"), d.get("end"), d["periods_per_year"])
    st = payload["strategy"]
    strategy = instantiate(StrategyRef(**st["code"]), st["options"], st["state"])
    params, combo_ids = payload["params"], payload["combo_ids"]
    T, N, C = len(market), len(market.symbols), len(params)
    tradable = tradable_mask(market)[:, :, None]
    asset_ret = market.returns.to_numpy()
    costs = payload["costs"]

    chunk = max(1, min(C, payload.get("max_cells", 20_000_000) // max(1, T * N)))
    rets = np.empty((T, C))
    scores: dict[str, list[np.ndarray]] = {k: [] for k in M.METRIC_COLUMNS}
    for s in range(0, C, chunk):
        W = strategy.positions_batch(market, params[s : s + chunk])
        if W.shape != (T, N, min(chunk, C - s)):
            raise ValueError(f"{type(strategy).__name__}.positions_batch returned {W.shape}, "
                             f"expected {(T, N, min(chunk, C - s))}")
        sim = simulate(np.where(tradable, W, 0.0), asset_ret, costs["cost_bps"], costs["delay"],
                       costs["borrow_bps"], d["periods_per_year"])
        rets[:, s : s + chunk] = sim.returns
        sc = M.score(sim.returns, d["periods_per_year"], sim.traded, sim.gross)
        for k in M.METRIC_COLUMNS:
            scores[k].append(sc[k])

    metrics_df = pd.DataFrame({"combo_id": combo_ids, **{k: np.concatenate(v) for k, v in scores.items()}})
    metrics_df = pd.concat([metrics_df, pd.DataFrame(params).add_prefix("p_")], axis=1)
    returns_df = pd.DataFrame(rets, index=market.index, columns=combo_ids)
    returns_df.index.name = "ts"

    m_path, r_path = output_paths(payload["results_dir"], payload["task_id"])
    _atomic_write(metrics_df, m_path)
    _atomic_write(returns_df, r_path)

    return {
        "task_id": payload["task_id"],
        "worker": f"{socket.gethostname()}:{os.getpid()}",
        "duration_s": time.perf_counter() - t0,
        "n_backtests": C,
        "best_sharpe": float(np.max(np.concatenate(scores["sharpe"]))),
    }

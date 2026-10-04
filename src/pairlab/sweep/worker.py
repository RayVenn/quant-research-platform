"""Task execution — runs on any worker (in-process, subprocess, or Ray node).

A task payload is fully self-describing (data location, pairs, parameters,
output location), so workers are stateless and any task can run anywhere.
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

from pairlab.data.store import cached_panel
from pairlab.engine import metrics as M
from pairlab.engine.vectorized import Thresholds, run_group

CHAOS_ENV = "PAIRLAB_CHAOS_FAIL_RATE"


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

    data = payload["data"]
    pairs = [tuple(p) for p in payload["pairs"]]
    symbols = tuple(sorted({s for p in pairs for s in p}))
    panel = cached_panel(data["store"], symbols, data.get("start"), data.get("end"))
    prices_y = panel[[y for y, _ in pairs]]
    prices_x = panel[[x for _, x in pairs]]

    combos = np.asarray(payload["combos"], dtype=float)
    th = Thresholds(entry=combos[:, 0], exit=combos[:, 1], stop=combos[:, 2])
    res = run_group(
        prices_y,
        prices_x,
        beta_window=payload["beta_window"],
        z_window=payload["z_window"],
        thresholds=th,
        cost_bps=payload["cost_bps"],
        delay=payload["delay"],
        periods_per_year=data["periods_per_year"],
    )

    P, C = len(pairs), len(th)
    combo_ids = payload["combo_ids"]
    metrics_df = pd.DataFrame(
        {
            "pair": np.repeat([f"{y}/{x}" for y, x in pairs], C),
            "combo_id": np.tile(combo_ids, P),
            "beta_window": payload["beta_window"],
            "z_window": payload["z_window"],
            "entry_z": np.tile(th.entry, P),
            "exit_z": np.tile(th.exit, P),
            "stop_z": np.tile(th.stop, P),
            **{k: res.pair_metrics[k].reshape(-1) for k in M.METRIC_COLUMNS},
        }
    )
    returns_df = pd.DataFrame(res.portfolio_returns, index=res.index, columns=combo_ids)
    returns_df.index.name = "ts"

    m_path, r_path = output_paths(payload["results_dir"], payload["task_id"])
    _atomic_write(metrics_df, m_path)
    _atomic_write(returns_df, r_path)

    return {
        "task_id": payload["task_id"],
        "worker": f"{socket.gethostname()}:{os.getpid()}",
        "duration_s": time.perf_counter() - t0,
        "n_backtests": P * C,
        "best_portfolio_sharpe": float(M.sharpe(res.portfolio_returns, data["periods_per_year"]).max()),
    }

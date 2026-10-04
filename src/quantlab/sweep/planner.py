"""Expand a strategy's parameter grid into deterministic, idempotent tasks."""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

from quantlab.strategy.base import Params, Strategy


def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def combo_id(params: Params) -> str:
    return ",".join(f"{k}={_fmt(v)}" for k, v in params.items())


def expand_grid(grid: dict[str, list[Any]], strategy: Strategy) -> list[Params]:
    keys = list(grid)
    combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]
    combos = [c for c in combos if strategy.valid(c)]
    if not combos:
        raise ValueError(f"strategy.grid produces no valid parameter combos for {type(strategy).__name__}")
    return combos


def plan_tasks(
    run_id: str, combos: list[Params], group_by: tuple[str, ...], combos_per_task: int, base: dict[str, Any]
) -> list[dict[str, Any]]:
    """One task = one value of the strategy's ``group_by`` params × a chunk of the other params.

    Grouping lets vectorized strategies compute shared state (rolling stats) once per task.
    ``task_id`` depends only on the run and the task's content, so re-planning is idempotent.
    """
    groups: dict[tuple, list[Params]] = {}
    for c in combos:
        groups.setdefault(tuple(c.get(k) for k in group_by), []).append(c)
    tasks = []
    for members in groups.values():
        for i in range(0, len(members), combos_per_task):
            chunk = members[i : i + combos_per_task]
            ids = [combo_id(c) for c in chunk]
            tid = hashlib.sha1(f"{run_id}|{'|'.join(ids)}".encode()).hexdigest()[:12]
            tasks.append({**base, "task_id": tid, "run_id": run_id, "params": chunk, "combo_ids": ids})
    return tasks

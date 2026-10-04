"""Expand a parameter grid into deterministic, idempotent tasks."""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

from pairlab.sweep.spec import GridSpec


def combo_id(beta_window: int, z_window: int, entry: float, exit_: float, stop: float) -> str:
    return f"b{beta_window}_z{z_window}_e{entry:g}_x{exit_:g}_s{stop:g}"


def threshold_combos(grid: GridSpec) -> list[tuple[float, float, float]]:
    """Valid (entry, exit, stop) triples: exit < entry < stop."""
    return [
        (e, x, s)
        for e, x, s in itertools.product(grid.entry_z, grid.exit_z, grid.stop_z)
        if x < e < s
    ]


def grid_size(grid: GridSpec) -> int:
    return len(grid.beta_window) * len(grid.z_window) * len(threshold_combos(grid))


def plan_tasks(run_id: str, grid: GridSpec, combos_per_task: int, base: dict[str, Any]) -> list[dict[str, Any]]:
    """One task = one (beta_window, z_window) group × a chunk of threshold combos × all pairs.

    Grouping by window keeps the rolling statistics computed once per task, and
    the threshold axis is what the engine vectorizes over. ``task_id`` depends
    only on the run and the task's content, so re-planning is idempotent.
    """
    combos = threshold_combos(grid)
    if not combos:
        raise ValueError("grid produces no valid (entry, exit, stop) combos; need exit < entry < stop")
    tasks = []
    for bw, zw in itertools.product(grid.beta_window, grid.z_window):
        for i in range(0, len(combos), combos_per_task):
            chunk = combos[i : i + combos_per_task]
            ids = [combo_id(bw, zw, *c) for c in chunk]
            tid = hashlib.sha1(f"{run_id}|{bw}|{zw}|{ids[0]}|{ids[-1]}|{len(ids)}".encode()).hexdigest()[:12]
            tasks.append(
                {
                    **base,
                    "task_id": tid,
                    "run_id": run_id,
                    "beta_window": bw,
                    "z_window": zw,
                    "combos": [list(c) for c in chunk],
                    "combo_ids": ids,
                }
            )
    return tasks

"""Sweep orchestrator: schedules tasks on an executor with retries, monitoring and crash recovery."""

from __future__ import annotations

import concurrent.futures.process as cfp
import json
import logging
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pairlab.sweep.executors import Executor, make_executor
from pairlab.sweep.spec import ExecutionSpec
from pairlab.sweep.state import StateStore
from pairlab.sweep.worker import execute_task, output_paths, outputs_exist

log = logging.getLogger("pairlab.sweep")


@dataclass
class Monitor:
    """Progress reporting: log lines, JSONL event stream and a Prometheus textfile."""

    run_id: str
    run_dir: Path
    total: int
    interval_s: float = 2.0
    started: float = field(default_factory=time.time)
    backtests_done: int = 0
    _last: float = 0.0

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"ts": time.time(), "run_id": self.run_id, "event": kind, **fields}
        with open(self.run_dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def tick(self, counts: dict[str, int], force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < self.interval_s:
            return
        self._last = now
        elapsed = max(now - self.started, 1e-9)
        done = counts["succeeded"] + counts["failed"]
        rate = self.backtests_done / elapsed
        remaining = self.total - done
        eta = f"{elapsed / done * remaining:.0f}s" if done else "?"
        log.info(
            "[%s] %d/%d tasks done (failed=%d running=%d) | %.0f backtests/s | ETA %s",
            self.run_id, counts["succeeded"], self.total, counts["failed"], counts["running"], rate, eta,
        )
        lines = [
            "# HELP pairlab_tasks Tasks by status",
            "# TYPE pairlab_tasks gauge",
            *(f'pairlab_tasks{{run_id="{self.run_id}",status="{k}"}} {v}' for k, v in counts.items()),
            "# TYPE pairlab_backtests_per_second gauge",
            f'pairlab_backtests_per_second{{run_id="{self.run_id}"}} {rate:.3f}',
            "# TYPE pairlab_backtests_total counter",
            f'pairlab_backtests_total{{run_id="{self.run_id}"}} {self.backtests_done}',
        ]
        tmp = self.run_dir / "metrics.prom.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(self.run_dir / "metrics.prom")


def recover(state: StateStore, run_id: str, results_dir: str, retry_failed: bool = False) -> dict[str, int]:
    """Make the state consistent after a crash.

    * ``running`` tasks whose outputs exist finished after the driver died → succeeded
    * other ``running`` tasks were lost with their worker/driver → pending
    * optionally give exhausted ``failed`` tasks a fresh retry budget
    """
    healed = requeued = 0
    for t in state.tasks(run_id, "running"):
        if outputs_exist(results_dir, t["task_id"]):
            state.mark_succeeded(t["task_id"], t["worker"], t["duration_s"])
            healed += 1
        else:
            state.conn.execute("UPDATE tasks SET status='pending' WHERE task_id=?", (t["task_id"],))
            requeued += 1
    reset_failed = state.reset(run_id, "failed", reset_attempts=True) if retry_failed else 0
    if healed or requeued or reset_failed:
        state.log_event(run_id, "warning", f"recovered: healed={healed} requeued={requeued} "
                                           f"reset_failed={reset_failed}")
    return {"healed": healed, "requeued": requeued, "reset_failed": reset_failed}


def run_tasks(
    state: StateStore,
    run_id: str,
    run_dir: Path,
    results_dir: str,
    execution: ExecutionSpec,
    executor: Executor | None = None,
) -> dict[str, int]:
    """Drive all pending tasks of *run_id* to a terminal state."""
    own_executor = executor is None
    ex = executor or make_executor(execution.backend, execution.max_workers, execution.ray_address)
    max_attempts = execution.max_retries + 1
    total = len(state.tasks(run_id))
    monitor = Monitor(run_id, run_dir, total)
    monitor.event("sweep_started", backend=ex.name, capacity=ex.capacity, total_tasks=total)
    state.set_run_status(run_id, "running")

    queue: deque[dict[str, Any]] = deque()
    for t in state.tasks(run_id, "pending"):
        if outputs_exist(results_dir, t["task_id"]):  # idempotency: never recompute finished work
            state.mark_succeeded(t["task_id"], None, None)
        else:
            queue.append(json.loads(t["payload"]))
    delayed: list[tuple[float, dict[str, Any]]] = []
    inflight: dict[Any, tuple[dict[str, Any], float]] = {}
    max_inflight = max(1, ex.capacity * 2)

    def fail(payload: dict[str, Any], err: str) -> None:
        status = state.mark_failed(payload["task_id"], err, max_attempts)
        level = "warning" if status == "pending" else "error"
        state.log_event(run_id, level, err.strip().splitlines()[-1], payload["task_id"])
        monitor.event("task_failed", task_id=payload["task_id"], attempt=payload["attempt"],
                      will_retry=status == "pending", error=err.strip().splitlines()[-1])
        if status == "pending":
            backoff = execution.retry_backoff_s * 2 ** (payload["attempt"] - 1)
            delayed.append((time.time() + backoff, payload))

    try:
        while queue or inflight or delayed:
            now = time.time()
            for item in [d for d in delayed if d[0] <= now]:
                delayed.remove(item)
                queue.append(item[1])

            while queue and len(inflight) < max_inflight:
                payload = queue.popleft()
                payload["attempt"] = state.mark_running(payload["task_id"])
                inflight[ex.submit(execute_task, payload)] = (payload, time.time())

            if not inflight:
                time.sleep(min(0.2, max(0.0, min((d[0] for d in delayed), default=now) - now)))
                continue

            for handle in ex.wait(list(inflight), timeout=0.5):
                payload, _ = inflight.pop(handle)
                try:
                    res = ex.result(handle)
                except cfp.BrokenProcessPool:
                    fail(payload, "worker process died (BrokenProcessPool)")
                    continue
                except Exception:  # noqa: BLE001 - any task error is retried
                    fail(payload, traceback.format_exc())
                    continue
                state.mark_succeeded(payload["task_id"], res["worker"], res["duration_s"])
                monitor.backtests_done += res["n_backtests"]
                monitor.event("task_succeeded", **res)

            if execution.task_timeout_s:
                for handle, (payload, started) in list(inflight.items()):
                    if time.time() - started > execution.task_timeout_s:
                        ex.cancel(handle)
                        inflight.pop(handle)
                        fail(payload, f"timeout after {execution.task_timeout_s}s")

            if ex.is_broken():
                for handle, (payload, _) in list(inflight.items()):
                    inflight.pop(handle)
                    fail(payload, "worker process died (BrokenProcessPool)")
                ex.shutdown()
                ex = make_executor(execution.backend, execution.max_workers, execution.ray_address)
                state.log_event(run_id, "warning", "executor restarted after worker crash")

            monitor.tick(state.counts(run_id))
    except KeyboardInterrupt:
        for payload, _ in inflight.values():
            state.conn.execute("UPDATE tasks SET status='pending' WHERE task_id=?", (payload["task_id"],))
        state.set_run_status(run_id, "interrupted")
        monitor.event("sweep_interrupted")
        raise
    finally:
        if own_executor:
            ex.shutdown()

    counts = state.counts(run_id)
    monitor.tick(counts, force=True)
    status = "swept" if counts["failed"] == 0 else "partial"
    state.set_run_status(run_id, status)
    monitor.event("sweep_finished", status=status, **counts)
    return counts


def aggregate(state: StateStore, run_id: str, results_dir: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate per-task outputs into run-level metrics and portfolio-returns tables."""
    metrics, returns = [], []
    for t in state.tasks(run_id, "succeeded"):
        m_path, r_path = output_paths(results_dir, t["task_id"])
        metrics.append(pd.read_parquet(m_path))
        returns.append(pd.read_parquet(r_path))
    if not metrics:
        raise RuntimeError(f"run {run_id} has no successful tasks to aggregate")
    m = pd.concat(metrics, ignore_index=True).sort_values(["combo_id", "pair"], ignore_index=True)
    r = pd.concat(returns, axis=1).sort_index(axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out_dir / "pair_metrics.parquet", index=False)
    r.to_parquet(out_dir / "portfolio_returns.parquet")
    return m, r

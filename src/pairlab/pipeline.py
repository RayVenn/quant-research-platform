"""End-to-end research pipeline: prepare → sweep → aggregate → validate → register."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pairlab import __version__
from pairlab.data.screening import screen_pairs
from pairlab.data.store import PriceStore, cached_panel, fingerprint
from pairlab.registry.store import Registry
from pairlab.sweep.executors import Executor
from pairlab.sweep.planner import grid_size, plan_tasks
from pairlab.sweep.runner import aggregate, recover, run_tasks
from pairlab.sweep.spec import JobSpec
from pairlab.sweep.state import StateStore
from pairlab.validation.report import validate

log = logging.getLogger("pairlab")


class Workspace:
    """Where runs, state and the registry live. Defaults to $PAIRLAB_HOME or the CWD."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("PAIRLAB_HOME", ".")).resolve()
        self.runs_dir = self.root / "runs"
        self.state_path = self.runs_dir / "state.db"
        self.registry = Registry(os.environ.get("PAIRLAB_REGISTRY", self.root / "registry"))

    def state(self) -> StateStore:
        return StateStore(self.state_path)


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    results_dir: str
    panel: pd.DataFrame
    pairs: list[tuple[str, str]]
    manifest: dict[str, Any]


def code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{__version__}+{sha}"
    except Exception:  # noqa: BLE001 - not a git checkout (e.g. container image)
        return os.environ.get("PAIRLAB_CODE_VERSION", __version__)


def resolve_pairs(spec: JobSpec, panel: pd.DataFrame) -> tuple[list[tuple[str, str]], list[dict]]:
    if spec.universe.pairs:
        return [tuple(p) for p in spec.universe.pairs], []
    sc = spec.universe.screen
    head = panel.iloc[: max(50, int(len(panel) * sc.train_fraction))]
    cands = screen_pairs(head, sc.min_corr, sc.max_pvalue, sc.max_half_life, sc.max_pairs)
    if not cands:
        raise RuntimeError("pair screen found no cointegrated pairs; relax universe.screen thresholds")
    return [(c.y, c.x) for c in cands], [c.to_dict() for c in cands]


def prepare(spec: JobSpec, ws: Workspace) -> RunContext:
    d = spec.data
    symbols = tuple(sorted(d.symbols or PriceStore(d.store).symbols()))
    universe = cached_panel(d.store, symbols, d.start, d.end)
    pairs, screened = resolve_pairs(spec, universe)
    pair_syms = tuple(sorted({s for p in pairs for s in p}))
    panel = cached_panel(d.store, pair_syms, d.start, d.end)
    data_fp = fingerprint(panel)
    digest = hashlib.sha256(f"{spec.research_hash()}|{data_fp}|{json.dumps(pairs)}".encode()).hexdigest()
    run_id = f"{spec.name}-{digest[:10]}"
    run_dir = ws.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "name": spec.name,
        "code_version": code_version(),
        "data_fingerprint": data_fp,
        "data_range": [str(panel.index[0]), str(panel.index[-1])],
        "n_bars": len(panel),
        "pairs": pairs,
        "screened": screened,
        "n_configs": grid_size(spec.grid),
        "n_backtests": grid_size(spec.grid) * len(pairs),
    }
    (run_dir / "spec.yaml").write_text(spec.to_yaml())
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return RunContext(run_id, run_dir, str(run_dir / "results"), panel, pairs, manifest)


def run_job(
    spec: JobSpec,
    ws: Workspace | None = None,
    retry_failed: bool = False,
    executor: Executor | None = None,
    expect_run_id: str | None = None,
) -> dict[str, Any]:
    ws = ws or Workspace()
    t0 = time.time()
    ctx = prepare(spec, ws)
    if expect_run_id and ctx.run_id != expect_run_id:
        raise RuntimeError(
            f"cannot resume {expect_run_id}: inputs changed (now resolves to {ctx.run_id}); "
            "data or spec differ from the original run"
        )
    state = ws.state()
    try:
        state.upsert_run(ctx.run_id, spec.name, ctx.manifest)
        if not state.tasks(ctx.run_id):
            base = {
                "data": {
                    "store": spec.data.store,
                    "start": spec.data.start,
                    "end": spec.data.end,
                    "periods_per_year": spec.data.periods_per_year,
                },
                "pairs": [list(p) for p in ctx.pairs],
                "cost_bps": spec.costs.cost_bps,
                "delay": spec.costs.delay,
                "results_dir": ctx.results_dir,
            }
            tasks = plan_tasks(ctx.run_id, spec.grid, spec.execution.combos_per_task, base)
            state.register_tasks(tasks)
            state.log_event(ctx.run_id, "info", f"planned {len(tasks)} tasks, "
                                                f"{ctx.manifest['n_backtests']} backtests")
        else:
            rec = recover(state, ctx.run_id, ctx.results_dir, retry_failed)
            log.info("resuming %s (%s)", ctx.run_id, rec)

        log.info("run %s: %d pairs × %d configs = %d backtests",
                 ctx.run_id, len(ctx.pairs), ctx.manifest["n_configs"], ctx.manifest["n_backtests"])
        counts = run_tasks(state, ctx.run_id, ctx.run_dir, ctx.results_dir, spec.execution, executor)
        summary: dict[str, Any] = {"run_id": ctx.run_id, "run_dir": str(ctx.run_dir), "tasks": counts}
        if counts["failed"]:
            log.error("run %s has %d failed tasks; fix and `pairlab resume %s --retry-failed`",
                      ctx.run_id, counts["failed"], ctx.run_id)
            summary["status"] = "partial"
            return summary

        pair_metrics, returns = aggregate(state, ctx.run_id, ctx.results_dir, ctx.run_dir / "aggregate")
        report = validate(spec, ctx.panel, ctx.pairs, pair_metrics, returns)
        report.pop("oos_equity").rename("equity").to_frame().to_parquet(ctx.run_dir / "oos_equity.parquet")
        (ctx.run_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str))

        artifact = {
            "name": spec.name,
            "version": ctx.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": ctx.manifest["code_version"],
            "data_fingerprint": ctx.manifest["data_fingerprint"],
            "data_range": ctx.manifest["data_range"],
            "spec": spec.model_dump(mode="json", exclude={"execution"}),
            "params": report["params"],
            "pairs": [p for p in report["pairs"] if p["selected"]],
            "validation": {k: report[k] for k in ("passed", "checks", "oos", "n_trials")},
        }
        ws.registry.register(artifact)
        status = "validated" if report["passed"] else "rejected"
        state.set_run_status(ctx.run_id, status)
        state.log_event(ctx.run_id, "info", f"validation {status}; registered {spec.name}:{ctx.run_id}")
        summary.update(status=status, validation=report, elapsed_s=round(time.time() - t0, 2))
        return summary
    finally:
        state.close()


def load_run_spec(ws: Workspace, run_id: str) -> JobSpec:
    p = ws.runs_dir / run_id / "spec.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no such run: {run_id}")
    return JobSpec.from_yaml(p)

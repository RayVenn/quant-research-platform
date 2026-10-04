"""End-to-end research pipeline: data → fit → audit → sweep → aggregate → validate → register."""

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

from quantlab import __version__
from quantlab.data.ingest import ingest
from quantlab.data.store import MarketData, PriceStore, cached_market, fingerprint
from quantlab.registry.store import Registry
from quantlab.strategy.audit import AuditReport, StrategyAuditError, audit
from quantlab.strategy.base import Params, Strategy
from quantlab.strategy.loader import StrategyRef, instantiate, resolve
from quantlab.sweep.executors import Executor
from quantlab.sweep.planner import combo_id, expand_grid, plan_tasks
from quantlab.sweep.runner import aggregate, recover, run_tasks
from quantlab.sweep.spec import JobSpec
from quantlab.sweep.state import StateStore
from quantlab.validation.report import validate

log = logging.getLogger("quantlab")


class Workspace:
    """Where runs, state and the registry live. Defaults to $QUANTLAB_HOME or the CWD."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.environ.get("QUANTLAB_HOME", ".")).resolve()
        self.runs_dir = self.root / "runs"
        self.state_path = self.runs_dir / "state.db"
        self.registry = Registry(os.environ.get("QUANTLAB_REGISTRY", self.root / "registry"))

    def state(self) -> StateStore:
        return StateStore(self.state_path)


@dataclass
class Resolved:
    sref: StrategyRef
    strategy: Strategy
    state: dict[str, Any]
    market: MarketData
    combos: list[Params]
    audit: AuditReport


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    results_dir: str
    r: Resolved
    manifest: dict[str, Any]


def code_version() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{__version__}+{sha}"
    except Exception:  # noqa: BLE001 - not a git checkout (e.g. container image)
        return os.environ.get("QUANTLAB_CODE_VERSION", __version__)


def ensure_data(spec: JobSpec) -> None:
    """Fetch any universe symbols missing from the store from the job's free data provider."""
    d = spec.data
    if not (d.provider and d.symbols):
        return
    res = ingest(d.provider, d.symbols, d.store, d.start, d.end, d.interval, **d.provider_options)
    if res["missing"]:
        raise RuntimeError(f"{d.provider} returned no data for {res['missing']}; fix data.symbols")


def resolve_job(spec: JobSpec) -> Resolved:
    """Load data, fit the strategy on the leading window, expand the grid, audit for lookahead."""
    ensure_data(spec)
    d = spec.data
    sref = resolve(spec.strategy.ref)
    universe_syms = tuple(sorted(d.symbols or PriceStore(d.store).symbols()))
    universe = cached_market(d.store, universe_syms, d.start, d.end, d.periods_per_year)
    fit_bars = max(50, int(len(universe) * spec.strategy.fit_fraction))
    raw_state = instantiate(sref, spec.strategy.options).fit(universe.slice(0, fit_bars)) or {}
    state = json.loads(json.dumps(raw_state, default=float))
    syms = tuple(sorted(state.get("symbols") or universe_syms))
    market = cached_market(d.store, syms, d.start, d.end, d.periods_per_year)
    strategy = instantiate(sref, spec.strategy.options, state)
    combos = expand_grid(spec.strategy.grid, strategy)
    rep = audit(strategy, market, combos[0])
    if not rep.ok:
        raise StrategyAuditError(f"{spec.strategy.ref} failed the strategy audit: " + "; ".join(rep.problems))
    return Resolved(sref, strategy, state, market, combos, rep)


def prepare(spec: JobSpec, ws: Workspace) -> RunContext:
    r = resolve_job(spec)
    data_fp = fingerprint(r.market)
    key = f"{spec.research_hash()}|{data_fp}|{r.sref.code_hash}|{json.dumps(r.state, sort_keys=True)}|{__version__}"
    run_id = f"{spec.name}-{hashlib.sha256(key.encode()).hexdigest()[:10]}"
    run_dir = ws.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "name": spec.name,
        "code_version": code_version(),
        "strategy": {"ref": r.sref.ref, "code_hash": r.sref.code_hash},
        "data_fingerprint": data_fp,
        "data_range": [str(r.market.index[0]), str(r.market.index[-1])],
        "n_bars": len(r.market),
        "symbols": r.market.symbols,
        "state": r.state,
        "n_configs": len(r.combos),
        "audit": r.audit.stats,
    }
    (run_dir / "spec.yaml").write_text(spec.to_yaml())
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if r.sref.source is not None:
        (run_dir / "strategy_source.py").write_text(r.sref.source)
    return RunContext(run_id, run_dir, str(run_dir / "results"), r, manifest)


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
    r = ctx.r
    if expect_run_id and ctx.run_id != expect_run_id:
        raise RuntimeError(
            f"cannot resume {expect_run_id}: inputs changed (now resolves to {ctx.run_id}); "
            "data, strategy code or spec differ from the original run"
        )
    state = ws.state()
    try:
        state.upsert_run(ctx.run_id, spec.name, ctx.manifest)
        if not state.tasks(ctx.run_id):
            d = spec.data
            base = {
                "data": {"store": d.store, "start": d.start, "end": d.end, "periods_per_year": d.periods_per_year,
                         "symbols": r.market.symbols},
                "strategy": {"code": r.sref.to_dict(), "options": spec.strategy.options, "state": r.state},
                "costs": spec.costs.model_dump(),
                "max_cells": spec.execution.max_cells,
                "results_dir": ctx.results_dir,
            }
            tasks = plan_tasks(ctx.run_id, r.combos, type(r.strategy).group_by, spec.execution.combos_per_task, base)
            state.register_tasks(tasks)
            state.log_event(ctx.run_id, "info", f"planned {len(tasks)} tasks, {len(r.combos)} backtests")
        else:
            rec = recover(state, ctx.run_id, ctx.results_dir, retry_failed)
            log.info("resuming %s (%s)", ctx.run_id, rec)

        log.info("run %s: strategy=%s, %d symbols × %d bars, %d param combos",
                 ctx.run_id, r.sref.ref, len(r.market.symbols), len(r.market), len(r.combos))
        counts = run_tasks(state, ctx.run_id, ctx.run_dir, ctx.results_dir, spec.execution, executor)
        summary: dict[str, Any] = {"run_id": ctx.run_id, "run_dir": str(ctx.run_dir), "tasks": counts}
        if counts["failed"]:
            log.error("run %s has %d failed tasks; fix and `quantlab resume %s --retry-failed`",
                      ctx.run_id, counts["failed"], ctx.run_id)
            summary["status"] = "partial"
            return summary

        _, returns = aggregate(state, ctx.run_id, ctx.results_dir, ctx.run_dir / "aggregate")
        report = validate(spec, r.market, r.strategy, {combo_id(c): c for c in r.combos}, returns)
        report.pop("oos_equity").rename("equity").to_frame().to_parquet(ctx.run_dir / "oos_equity.parquet")
        (ctx.run_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str))

        d = spec.data
        artifact = {
            "name": spec.name,
            "version": ctx.run_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": ctx.manifest["code_version"],
            "strategy": {**r.sref.to_dict(), "options": spec.strategy.options, "state": r.state},
            "params": report["params"],
            "costs": spec.costs.model_dump(),
            "data": {"store": d.store, "provider": d.provider, "provider_options": d.provider_options,
                     "symbols": r.market.symbols, "start": d.start, "interval": d.interval,
                     "periods_per_year": d.periods_per_year, "fingerprint": ctx.manifest["data_fingerprint"],
                     "range": ctx.manifest["data_range"]},
            "spec": spec.model_dump(mode="json", exclude={"execution"}),
            "validation": {k: report[k] for k in ("passed", "checks", "oos", "n_trials", "n_effective_trials")},
            "snapshot": report["snapshot"],
        }
        ws.registry.register(json.loads(json.dumps(artifact, default=str)))
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

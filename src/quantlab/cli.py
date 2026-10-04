"""``quantlab`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from quantlab.data import providers
from quantlab.data.ingest import ingest
from quantlab.data.store import PriceStore
from quantlab.data.synthetic import make_universe
from quantlab.live import target_weights
from quantlab.pipeline import Workspace, load_run_spec, resolve_job, run_job
from quantlab.registry.store import RegistryError
from quantlab.strategy import loader, template
from quantlab.strategy.audit import StrategyAuditError, audit
from quantlab.sweep.spec import JobSpec


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "(none)"
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    body = ["  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([head, "  ".join("-" * widths[c] for c in cols), *body])


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


# ---------------------------------------------------------------- data
def cmd_data_synth(a) -> int:
    panel = make_universe(a.clusters, a.per_cluster, a.noise, a.bars, seed=a.seed)
    PriceStore(a.store).write_panel(panel)
    print(f"wrote {panel.shape[1]} symbols × {len(panel)} bars to {a.store}")
    return 0


def cmd_data_import(a) -> int:
    bars = providers.get_provider("csv", path=a.csv, format=a.format, ts_col=a.ts_col).fetch()
    for sym, df in bars.items():
        PriceStore(a.store).write(sym, df, {"provider": "csv", "source": a.csv})
    print(f"imported {len(bars)} symbols into {a.store}")
    return 0


def _symbols(a) -> list[str]:
    syms = list(a.symbols or [])
    if a.symbols_file:
        syms += [ln.split("#")[0].strip() for ln in Path(a.symbols_file).read_text().splitlines()]
    syms = [s for s in dict.fromkeys(syms) if s]
    if not syms:
        raise SystemExit("give --symbols and/or --symbols-file")
    return syms


def cmd_data_fetch(a) -> int:
    res = ingest(a.provider, _symbols(a), a.store, a.start, a.end, a.interval, refresh=a.refresh)
    print(f"fetched={len(res['fetched'])} already_present={len(res['skipped'])} missing={res['missing']}")
    return 1 if res["missing"] else 0


def cmd_data_list(a) -> int:
    ps = PriceStore(a.store)
    rows = []
    for sym in ps.symbols():
        m = ps.meta(sym)
        if not m:
            df = ps.read(sym)
            m = {"first": str(df.index[0]), "last": str(df.index[-1]), "bars": len(df)}
        rows.append({"symbol": sym, "provider": m.get("provider", "?"), "first": str(m.get("first"))[:10],
                     "last": str(m.get("last"))[:10], "bars": m.get("bars"), "fetched_at": m.get("fetched_at")})
    print(_table(rows, ["symbol", "provider", "first", "last", "bars", "fetched_at"]))
    return 0


def cmd_data_providers(a) -> int:
    print(_table([{"name": k, "target": v} for k, v in providers.available().items()], ["name", "target"]))
    return 0


# ---------------------------------------------------------------- strategies
def cmd_strategy_list(a) -> int:
    rows = []
    for name, target in loader.available().items():
        cls = loader.load_class(loader.resolve(name))
        rows.append({"name": name, "target": target, "group_by": ",".join(cls.group_by),
                     "summary": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""})
    print(_table(rows, ["name", "target", "group_by", "summary"]))
    return 0


def cmd_strategy_new(a) -> int:
    name = a.name
    cls = "".join(w.capitalize() for w in name.split("_"))
    title = name.replace("_", " ").title()
    spath = Path(a.dir) / f"{name}.py"
    jpath = Path(a.jobs_dir) / f"{name}.yaml"
    for p in (spath, jpath):
        if p.exists():
            raise SystemExit(f"{p} already exists")
        p.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(template.STRATEGY.format(name=name, cls=cls, title=title))
    jpath.write_text(template.JOB.format(name=name, title=title, path=spath.as_posix()))
    print(f"created {spath} and {jpath}\nnext: quantlab strategy check {jpath} && quantlab run {jpath}")
    return 0


def cmd_strategy_check(a) -> int:
    spec = JobSpec.from_yaml(a.job)
    try:
        r = resolve_job(spec)
    except StrategyAuditError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    reports = [audit(r.strategy, r.market, c) for c in (r.combos[0], r.combos[-1])]
    problems = [p for rep in reports for p in rep.problems]
    out = {
        "strategy": r.sref.ref, "code_hash": r.sref.code_hash,
        "symbols": len(r.market.symbols), "bars": len(r.market),
        "range": [str(r.market.index[0].date()), str(r.market.index[-1].date())],
        "param_combos": len(r.combos), "group_by": list(type(r.strategy).group_by),
        "fit_state": {k: v for k, v in r.state.items() if k != "screened"},
        "audit": {"passed": not problems, "problems": problems, **reports[0].stats},
    }
    print(json.dumps(out, indent=2, default=str))
    return 1 if problems else 0


# ---------------------------------------------------------------- runs
def _apply_overrides(spec: JobSpec, a) -> JobSpec:
    ex = spec.execution.model_copy(update={
        k: v for k, v in {"backend": a.backend, "max_workers": a.workers, "ray_address": a.ray_address}.items()
        if v is not None
    })
    return spec.model_copy(update={"execution": ex})


def _print_summary(s: dict) -> int:
    print(f"\nrun {s['run_id']}  status={s['status']}  tasks={s['tasks']}")
    if "validation" in s:
        v = s["validation"]
        print(f"best params: {v['params']}")
        print(_table([{"check": k, **c} for k, c in v["checks"].items()], ["check", "value", "rule", "passed"]))
        print(f"\n{'PASSED' if v['passed'] else 'REJECTED'} — artifact registered as "
              f"{s['run_id'].rsplit('-', 1)[0]}:{s['run_id']} (stage: research)")
    return 0 if s["status"] in ("validated", "rejected") else 2


def cmd_run(a) -> int:
    spec = _apply_overrides(JobSpec.from_yaml(a.job), a)
    return _print_summary(run_job(spec, Workspace(a.home), retry_failed=a.retry_failed))


def cmd_resume(a) -> int:
    ws = Workspace(a.home)
    spec = _apply_overrides(load_run_spec(ws, a.run_id), a)
    return _print_summary(run_job(spec, ws, retry_failed=a.retry_failed, expect_run_id=a.run_id))


def _latest_run(ws: Workspace) -> str:
    st = ws.state()
    row = st.conn.execute("SELECT run_id FROM runs ORDER BY updated_at DESC LIMIT 1").fetchone()
    st.close()
    if not row:
        raise SystemExit("no runs yet")
    return row[0]


def cmd_status(a) -> int:
    ws = Workspace(a.home)
    run_id = a.run_id or _latest_run(ws)
    st = ws.state()
    run = st.get_run(run_id)
    if not run:
        raise SystemExit(f"no such run: {run_id}")
    counts = st.counts(run_id)
    tasks = st.tasks(run_id)
    durs = np.array([t["duration_s"] for t in tasks if t["duration_s"] is not None])
    failed = [t for t in tasks if t["status"] == "failed" or (t["error"] and t["status"] != "succeeded")]
    out = {
        "run_id": run_id,
        "status": run["status"],
        "updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run["updated_at"])),
        "tasks": counts,
        "retried_tasks": sum(1 for t in tasks if t["attempts"] > 1),
        "task_duration_s": {
            "p50": float(np.percentile(durs, 50)) if len(durs) else None,
            "p95": float(np.percentile(durs, 95)) if len(durs) else None,
        },
        "workers": sorted({t["worker"].split(":")[0] for t in tasks if t["worker"]}),
        "errors": [{"task_id": t["task_id"], "attempts": t["attempts"], "error": (t["error"] or "").strip()
                    .splitlines()[-1:]} for t in failed][:10],
        "recent_events": [{"level": e["level"], "task": e["task_id"], "msg": e["message"]}
                          for e in st.events(run_id, 10)],
    }
    st.close()
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_runs(a) -> int:
    st = Workspace(a.home).state()
    rows = [dict(r) for r in st.conn.execute("SELECT run_id, name, status, updated_at FROM runs ORDER BY updated_at")]
    for r in rows:
        r.update(st.counts(r["run_id"]))
        r["updated_at"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["updated_at"]))
    st.close()
    print(_table(rows, ["run_id", "status", "succeeded", "failed", "pending", "updated_at"]))
    return 0


# ---------------------------------------------------------------- registry
def cmd_registry(a) -> int:
    reg = Workspace(a.home).registry
    try:
        if a.action == "list":
            print(_table(reg.list(), ["name", "version", "stage", "passed", "oos_sharpe", "created_at"]))
        elif a.action == "show":
            print(json.dumps(reg.get(a.name, a.version, a.stage or (None if a.version else "production")), indent=2))
        elif a.action == "promote":
            reg.promote(a.name, a.version, a.stage, force=a.force, note=a.note or "")
            print(f"promoted {a.name}:{a.version} → {a.stage}")
        elif a.action == "history":
            for h in reg.history(a.name):
                print(json.dumps(h))
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_signal(a) -> int:
    reg = Workspace(a.home).registry
    try:
        art = reg.get(a.name, a.version, None if a.version else a.stage)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(target_weights(art, refresh=a.refresh, store=a.store), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantlab", description=__doc__)
    p.add_argument("--home", help="workspace root (default: $QUANTLAB_HOME or CWD)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data", help="manage the price store").add_subparsers(dest="data_cmd", required=True)
    s = data.add_parser("synth", help="write a synthetic universe with known cointegrated clusters (offline demo)")
    s.add_argument("--store", required=True)
    s.add_argument("--clusters", type=int, default=4)
    s.add_argument("--per-cluster", type=int, default=3)
    s.add_argument("--noise", type=int, default=4)
    s.add_argument("--bars", type=int, default=2520)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_data_synth)
    s = data.add_parser("import-csv", help="import bars from CSV (wide: ts,SYM1,SYM2… | long: ts,symbol,close,…)")
    s.add_argument("csv")
    s.add_argument("--store", required=True)
    s.add_argument("--format", choices=["wide", "long"], default="wide")
    s.add_argument("--ts-col", default="ts")
    s.set_defaults(fn=cmd_data_import)
    s = data.add_parser("fetch", help="download bars from a data provider (default: Yahoo Finance, free)")
    s.add_argument("--provider", default="yahoo")
    s.add_argument("--symbols", nargs="*")
    s.add_argument("--symbols-file", help="one symbol per line, '#' comments")
    s.add_argument("--store", required=True)
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--interval", default="1d")
    s.add_argument("--refresh", action="store_true", help="re-download symbols already in the store")
    s.set_defaults(fn=cmd_data_fetch)
    s = data.add_parser("list", help="symbols in a store with coverage and provenance")
    s.add_argument("--store", required=True)
    s.set_defaults(fn=cmd_data_list)
    data.add_parser("providers", help="available data providers").set_defaults(fn=cmd_data_providers)

    strat = sub.add_parser("strategy", help="strategy plugins").add_subparsers(dest="strategy_cmd", required=True)
    strat.add_parser("list", help="built-in and installed strategies").set_defaults(fn=cmd_strategy_list)
    s = strat.add_parser("new", help="scaffold a strategy plugin + job file")
    s.add_argument("name")
    s.add_argument("--dir", default="strategies")
    s.add_argument("--jobs-dir", default="jobs")
    s.set_defaults(fn=cmd_strategy_new)
    s = strat.add_parser("check", help="load data, fit, and audit a job's strategy for lookahead")
    s.add_argument("job")
    s.set_defaults(fn=cmd_strategy_check)

    for name, fn, helptext in (("run", cmd_run, "run a job spec end to end"),
                               ("resume", cmd_resume, "resume an interrupted/partial run")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("run_id" if name == "resume" else "job")
        s.add_argument("--backend", choices=["local", "process", "ray"])
        s.add_argument("--workers", type=int)
        s.add_argument("--ray-address")
        s.add_argument("--retry-failed", action="store_true", help="give exhausted tasks a fresh retry budget")
        s.set_defaults(fn=fn)

    s = sub.add_parser("status", help="progress, failures and recent events for a run")
    s.add_argument("run_id", nargs="?")
    s.set_defaults(fn=cmd_status)
    sub.add_parser("runs", help="list runs").set_defaults(fn=cmd_runs)

    reg = sub.add_parser("registry", help="strategy registry").add_subparsers(dest="action", required=True)
    reg.add_parser("list").set_defaults(fn=cmd_registry)
    s = reg.add_parser("show")
    s.add_argument("name")
    s.add_argument("--version")
    s.add_argument("--stage", choices=["staging", "production"])
    s.set_defaults(fn=cmd_registry)
    s = reg.add_parser("promote")
    s.add_argument("name")
    s.add_argument("version")
    s.add_argument("--stage", required=True, choices=["staging", "production"])
    s.add_argument("--force", action="store_true")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_registry)
    s = reg.add_parser("history")
    s.add_argument("name")
    s.set_defaults(fn=cmd_registry)

    s = sub.add_parser("signal", help="current target weights from a registered strategy (production hand-off)")
    s.add_argument("name")
    s.add_argument("--stage", default="production", choices=["research", "staging", "production"])
    s.add_argument("--version")
    s.add_argument("--refresh", action="store_true", help="pull latest bars from the provider first")
    s.add_argument("--store", help="override the artifact's data store")
    s.set_defaults(fn=cmd_signal)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

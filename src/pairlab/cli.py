"""``pairlab`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

from pairlab.data.screening import screen_pairs
from pairlab.data.store import PriceStore, cached_panel
from pairlab.data.synthetic import make_universe
from pairlab.pipeline import Workspace, load_run_spec, run_job
from pairlab.registry.store import RegistryError
from pairlab.sweep.spec import JobSpec


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
    df = pd.read_csv(a.csv, parse_dates=[a.ts_col])
    if a.format == "long":
        df = df.pivot(index=a.ts_col, columns="symbol", values="close")
    else:
        df = df.set_index(a.ts_col)
    PriceStore(a.store).write_panel(df)
    print(f"imported {df.shape[1]} symbols × {len(df)} bars into {a.store}")
    return 0


def cmd_screen(a) -> int:
    spec = JobSpec.from_yaml(a.job)
    d, sc = spec.data, spec.universe.screen or JobSpec.model_fields["universe"].default.screen
    store = PriceStore(d.store)
    panel = cached_panel(d.store, tuple(sorted(d.symbols or store.symbols())), d.start, d.end)
    head = panel.iloc[: max(50, int(len(panel) * sc.train_fraction))]
    cands = screen_pairs(head, sc.min_corr, sc.max_pvalue, sc.max_half_life, sc.max_pairs)
    print(_table([c.to_dict() for c in cands], ["y", "x", "corr", "pvalue", "hedge_ratio", "half_life"]))
    return 0


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
        print(f"params: {v['params']}")
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pairlab", description=__doc__)
    p.add_argument("--home", help="workspace root (default: $PAIRLAB_HOME or CWD)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data", help="manage the price store").add_subparsers(dest="data_cmd", required=True)
    s = data.add_parser("synth", help="write a synthetic universe with known cointegrated clusters")
    s.add_argument("--store", required=True)
    s.add_argument("--clusters", type=int, default=4)
    s.add_argument("--per-cluster", type=int, default=3)
    s.add_argument("--noise", type=int, default=4)
    s.add_argument("--bars", type=int, default=2520)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_data_synth)
    s = data.add_parser("import-csv", help="import closes from CSV (wide: ts,SYM1,SYM2… | long: ts,symbol,close)")
    s.add_argument("csv")
    s.add_argument("--store", required=True)
    s.add_argument("--format", choices=["wide", "long"], default="wide")
    s.add_argument("--ts-col", default="ts")
    s.set_defaults(fn=cmd_data_import)

    s = sub.add_parser("screen", help="screen the job's universe for cointegrated pairs")
    s.add_argument("job")
    s.set_defaults(fn=cmd_screen)

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

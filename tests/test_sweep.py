import json

import pandas as pd
import pytest

from quantlab.pipeline import prepare, run_job
from quantlab.strategies.pairs import PairsStrategy
from quantlab.sweep.planner import expand_grid, plan_tasks
from quantlab.sweep.runner import recover
from quantlab.sweep.worker import CHAOS_ENV

GRID_SIZE = 8


def test_planner_is_deterministic_and_complete(spec):
    combos = expand_grid(spec.strategy.grid, PairsStrategy())
    assert len(combos) == GRID_SIZE and all(c["exit_z"] < c["entry_z"] < c["stop_z"] for c in combos)
    a = plan_tasks("r1", combos, PairsStrategy.group_by, 3, {})
    b = plan_tasks("r1", combos, PairsStrategy.group_by, 3, {})
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert len({c for t in a for c in t["combo_ids"]}) == GRID_SIZE
    for t in a:  # every task shares one (beta_window, z_window) group
        assert len({(p["beta_window"], p["z_window"]) for p in t["params"]}) == 1


def test_run_job_end_to_end(spec, workspace):
    s = run_job(spec, workspace)
    assert s["status"] in ("validated", "rejected")
    assert s["tasks"]["succeeded"] == 4 and s["tasks"]["failed"] == 0
    run_dir = workspace.runs_dir / s["run_id"]
    m = pd.read_parquet(run_dir / "aggregate" / "combo_metrics.parquet")
    assert len(m) == GRID_SIZE and {"sharpe", "turnover", "p_entry_z"} <= set(m.columns)
    report = json.loads((run_dir / "validation.json").read_text())
    assert set(report["checks"]) == {"oos_sharpe", "deflated_sharpe", "pbo", "oos_max_drawdown", "n_stable_pairs"}
    art = workspace.registry.get(spec.name, s["run_id"])
    assert art["params"] == report["params"] and art["strategy"]["ref"].endswith("PairsStrategy")
    assert len(art["snapshot"]["pairs"]) == 4


def test_memory_chunking_does_not_change_results(spec, workspace, tmp_path):
    a = run_job(spec, workspace)
    ra = pd.read_parquet(workspace.runs_dir / a["run_id"] / "aggregate" / "returns.parquet")
    from quantlab.pipeline import Workspace

    ws2 = Workspace(tmp_path / "other")
    ex = spec.execution.model_copy(update={"max_cells": 1})
    b = run_job(spec.model_copy(update={"execution": ex}), ws2)
    rb = pd.read_parquet(ws2.runs_dir / b["run_id"] / "aggregate" / "returns.parquet")
    pd.testing.assert_frame_equal(ra, rb)


def test_strategy_code_change_changes_run_id(spec, workspace, tmp_path):
    src = (
        "from quantlab.strategies.trend import MovingAverageCrossover\n"
        "class Mine(MovingAverageCrossover):\n    name = 'mine'\n"
    )
    f = tmp_path / "mine.py"
    f.write_text(src)
    strat = {"ref": str(f), "grid": {"fast": [10], "slow": [50]}}
    s1 = spec.model_copy(update={"strategy": spec.strategy.model_validate(strat)})
    r1 = prepare(s1, workspace).run_id
    f.write_text(src + "# edited\n")
    assert prepare(s1, workspace).run_id != r1


def test_rerun_is_idempotent(spec, workspace):
    first = run_job(spec, workspace)
    st = workspace.state()
    before = {t["task_id"]: t["finished_at"] for t in st.tasks(first["run_id"])}
    second = run_job(spec, workspace)
    after = {t["task_id"]: t["finished_at"] for t in st.tasks(second["run_id"])}
    st.close()
    assert first["run_id"] == second["run_id"]
    assert before == after  # nothing recomputed


def test_execution_settings_do_not_change_run_id(spec, workspace):
    other = spec.model_copy(update={"execution": spec.execution.model_copy(update={"backend": "process"})})
    assert prepare(spec, workspace).run_id == prepare(other, workspace).run_id


def test_transient_failures_are_retried(spec, workspace, monkeypatch):
    monkeypatch.setenv(CHAOS_ENV, "0.5")
    ex = spec.execution.model_copy(update={"max_retries": 8})
    s = run_job(spec.model_copy(update={"execution": ex}), workspace)
    st = workspace.state()
    tasks = st.tasks(s["run_id"])
    st.close()
    assert s["tasks"]["failed"] == 0
    assert any(t["attempts"] > 1 for t in tasks)


def test_exhausted_failures_then_resume(spec, workspace, monkeypatch):
    monkeypatch.setenv(CHAOS_ENV, "1.0")
    s = run_job(spec, workspace)
    assert s["status"] == "partial" and s["tasks"]["failed"] == 4
    monkeypatch.setenv(CHAOS_ENV, "0")
    still = run_job(spec, workspace)
    assert still["status"] == "partial"  # failed tasks keep their exhausted budget...
    healed = run_job(spec, workspace, retry_failed=True)  # ...until explicitly retried
    assert healed["status"] in ("validated", "rejected") and healed["tasks"]["failed"] == 0


def test_crash_recovery(spec, workspace):
    s = run_job(spec, workspace)
    st = workspace.state()
    tasks = st.tasks(s["run_id"])
    results_dir = str(workspace.runs_dir / s["run_id"] / "results")
    # Simulate a driver crash: two tasks left 'running'; one of them never wrote its outputs.
    lost, finished = tasks[0]["task_id"], tasks[1]["task_id"]
    st.conn.execute("UPDATE tasks SET status='running' WHERE task_id IN (?, ?)", (lost, finished))
    for suffix in ("metrics", "returns"):
        (workspace.runs_dir / s["run_id"] / "results" / f"{lost}.{suffix}.parquet").unlink()
    assert recover(st, s["run_id"], results_dir) == {"healed": 1, "requeued": 1, "reset_failed": 0}
    st.close()
    again = run_job(spec, workspace)
    assert again["tasks"]["succeeded"] == len(tasks)


def test_resume_rejects_changed_inputs(spec, workspace):
    s = run_job(spec, workspace)
    changed = spec.model_copy(update={"costs": spec.costs.model_copy(update={"cost_bps": 9.0})})
    with pytest.raises(RuntimeError, match="inputs changed"):
        run_job(changed, workspace, expect_run_id=s["run_id"])


def test_process_backend(spec, workspace):
    ex = spec.execution.model_copy(update={"backend": "process", "max_workers": 2})
    s = run_job(spec.model_copy(update={"execution": ex}), workspace)
    assert s["tasks"]["succeeded"] == 4


def test_user_file_plugin_on_process_workers(spec, workspace, tmp_path):
    f = tmp_path / "rev.py"
    f.write_text(
        "import numpy as np\nfrom quantlab.strategy import Strategy\n\n"
        "class Reversal(Strategy):\n"
        "    group_by = ('lookback',)\n"
        "    def warmup(self, p):\n        return p['lookback']\n"
        "    def positions(self, data, p):\n"
        "        r = data.close.pct_change(p['lookback'])\n"
        "        return -np.sign(r).fillna(0) / data.close.shape[1]\n"
    )
    job = spec.model_copy(update={
        "strategy": spec.strategy.model_validate({"ref": str(f), "grid": {"lookback": [5, 10, 20]}}),
        "execution": spec.execution.model_copy(update={"backend": "process", "max_workers": 2}),
    })
    s = run_job(job, workspace)
    assert s["tasks"]["succeeded"] == 3 and s["status"] in ("validated", "rejected")
    assert (workspace.runs_dir / s["run_id"] / "strategy_source.py").read_text() == f.read_text()


def test_lookahead_strategy_is_rejected_before_sweep(spec, workspace, tmp_path):
    from quantlab.strategy.audit import StrategyAuditError

    f = tmp_path / "peek.py"
    f.write_text(
        "from quantlab.strategy import Strategy\n\n"
        "class Peek(Strategy):\n"
        "    def positions(self, data, p):\n"
        "        return (data.close.shift(-1) > data.close).astype(float) / data.close.shape[1]\n"
    )
    job = spec.model_copy(update={"strategy": spec.strategy.model_validate({"ref": str(f), "grid": {"x": [1]}})})
    with pytest.raises(StrategyAuditError, match="lookahead"):
        run_job(job, workspace)

import json

import pandas as pd
import pytest

from pairlab.pipeline import prepare, run_job
from pairlab.sweep.planner import grid_size, plan_tasks, threshold_combos
from pairlab.sweep.runner import recover
from pairlab.sweep.worker import CHAOS_ENV


def test_planner_is_deterministic_and_complete(spec):
    combos = threshold_combos(spec.grid)
    assert all(x < e < s for e, x, s in combos)
    a = plan_tasks("r1", spec.grid, 3, {})
    b = plan_tasks("r1", spec.grid, 3, {})
    assert [t["task_id"] for t in a] == [t["task_id"] for t in b]
    assert sum(len(t["combo_ids"]) for t in a) == grid_size(spec.grid)
    assert len({c for t in a for c in t["combo_ids"]}) == grid_size(spec.grid)


def test_run_job_end_to_end(spec, workspace):
    s = run_job(spec, workspace)
    assert s["status"] in ("validated", "rejected")
    assert s["tasks"]["succeeded"] == 4 and s["tasks"]["failed"] == 0
    run_dir = workspace.runs_dir / s["run_id"]
    m = pd.read_parquet(run_dir / "aggregate" / "pair_metrics.parquet")
    assert len(m) == 4 * grid_size(spec.grid)
    report = json.loads((run_dir / "validation.json").read_text())
    assert set(report["checks"]) == {"oos_sharpe", "deflated_sharpe", "pbo", "oos_max_drawdown", "n_stable_pairs"}
    assert workspace.registry.get(spec.name, s["run_id"])["params"]["combo_id"] == report["params"]["combo_id"]


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

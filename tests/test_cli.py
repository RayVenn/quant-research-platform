import json
from pathlib import Path

import yaml

from quantlab.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    home = str(tmp_path)
    store = str(tmp_path / "prices")
    assert main(["--home", home, "data", "synth", "--store", store, "--bars", "1500", "--clusters", "3"]) == 0
    job = yaml.safe_load(Path("jobs/demo.yaml").read_text())
    job["data"]["store"] = store
    job["execution"].update(backend="local")
    job["strategy"]["grid"].update(beta_window=[120], z_window=[20], entry_z=[1.5, 2.0], exit_z=[0.0], stop_z=[4.0])
    job["validation"].update(train_bars=400, test_bars=200)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(job))

    assert main(["--home", home, "strategy", "check", str(path)]) == 0
    assert main(["--home", home, "run", str(path)]) == 0
    assert main(["--home", home, "status"]) == 0
    assert main(["--home", home, "runs"]) == 0
    assert main(["--home", home, "registry", "list"]) == 0
    out = capsys.readouterr().out
    assert "demo_pairs" in out and "succeeded" in out

    version = next(r["version"] for r in _registry(home))
    assert main(["--home", home, "signal", "demo_pairs", "--version", version]) == 0
    sig = json.loads(capsys.readouterr().out)
    assert sig["version"] == version and sig["warnings"] == [] and "weights" in sig


def _registry(home):
    from quantlab.pipeline import Workspace

    return Workspace(home).registry.list()


def test_scaffolded_strategy_passes_check_and_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["data", "synth", "--store", "prices", "--bars", "1500"]) == 0
    assert main(["strategy", "new", "my_reversion"]) == 0
    job = yaml.safe_load(Path("jobs/my_reversion.yaml").read_text())
    job["data"] = {"store": "prices"}  # offline: synthetic store instead of Yahoo
    job["execution"] = {"backend": "local"}
    job["validation"] = {"train_bars": 400, "test_bars": 200}
    Path("jobs/my_reversion.yaml").write_text(yaml.safe_dump(job))
    assert main(["strategy", "check", "jobs/my_reversion.yaml"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("\n{") + 1 :])["audit"]["passed"]
    assert main(["run", "jobs/my_reversion.yaml"]) == 0
    assert main(["strategy", "list"]) == 0
    assert "momentum" in capsys.readouterr().out

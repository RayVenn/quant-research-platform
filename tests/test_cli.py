from pathlib import Path

import yaml

from pairlab.cli import main


def test_cli_end_to_end(tmp_path, capsys):
    home = str(tmp_path)
    store = str(tmp_path / "prices")
    assert main(["--home", home, "data", "synth", "--store", store, "--bars", "1500", "--clusters", "3"]) == 0
    job = yaml.safe_load(Path("jobs/demo.yaml").read_text())
    job["data"]["store"] = store
    job["execution"].update(backend="local")
    job["grid"].update(beta_window=[120], z_window=[20], entry_z=[1.5, 2.0], exit_z=[0.0], stop_z=[4.0])
    job["validation"].update(train_bars=400, test_bars=200)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(job))

    assert main(["--home", home, "screen", str(path)]) == 0
    assert main(["--home", home, "run", str(path)]) == 0
    assert main(["--home", home, "status"]) == 0
    assert main(["--home", home, "runs"]) == 0
    assert main(["--home", home, "registry", "list"]) == 0
    out = capsys.readouterr().out
    assert "demo_pairs" in out and "succeeded" in out

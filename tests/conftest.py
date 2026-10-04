from __future__ import annotations

import pytest

from pairlab.data.store import PriceStore, cached_panel
from pairlab.data.synthetic import make_universe
from pairlab.pipeline import Workspace
from pairlab.sweep.spec import JobSpec


@pytest.fixture(scope="session")
def panel():
    return make_universe(n_clusters=3, per_cluster=2, n_noise=2, n_bars=1500, seed=3)


@pytest.fixture
def workspace(tmp_path, panel):
    PriceStore(str(tmp_path / "prices")).write_panel(panel)
    cached_panel.cache_clear()
    return Workspace(tmp_path)


@pytest.fixture
def spec(workspace) -> JobSpec:
    return JobSpec.model_validate(
        {
            "name": "test_job",
            "data": {"store": str(workspace.root / "prices")},
            "universe": {"pairs": [["C0M0", "C0M1"], ["C1M0", "C1M1"], ["C2M0", "C2M1"], ["N0", "N1"]]},
            "grid": {
                "beta_window": [60, 120],
                "z_window": [20],
                "entry_z": [1.5, 2.0],
                "exit_z": [0.0, 0.5],
                "stop_z": [4.0],
            },
            "execution": {"backend": "local", "combos_per_task": 2, "max_retries": 2, "retry_backoff_s": 0.0},
            "validation": {"train_bars": 400, "test_bars": 200, "pbo_splits": 6},
        }
    )

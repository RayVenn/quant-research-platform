from __future__ import annotations

import pytest

from quantlab.data.store import MarketData, PriceStore, cached_market
from quantlab.data.synthetic import make_universe
from quantlab.pipeline import Workspace
from quantlab.sweep.spec import JobSpec


@pytest.fixture(scope="session")
def panel():
    return make_universe(n_clusters=3, per_cluster=2, n_noise=2, n_bars=1500, seed=3)


@pytest.fixture(scope="session")
def market(panel) -> MarketData:
    return MarketData(panel)


@pytest.fixture
def workspace(tmp_path, panel):
    PriceStore(str(tmp_path / "prices")).write_panel(panel)
    cached_market.cache_clear()
    return Workspace(tmp_path)


@pytest.fixture
def spec(workspace) -> JobSpec:
    return JobSpec.model_validate(
        {
            "name": "test_job",
            "data": {"store": str(workspace.root / "prices")},
            "strategy": {
                "ref": "pairs",
                "options": {"pairs": [["C0M0", "C0M1"], ["C1M0", "C1M1"], ["C2M0", "C2M1"], ["N0", "N1"]]},
                "grid": {
                    "beta_window": [60, 120],
                    "z_window": [20],
                    "entry_z": [1.5, 2.0],
                    "exit_z": [0.0, 0.5],
                    "stop_z": [4.0],
                },
            },
            "execution": {"backend": "local", "combos_per_task": 2, "max_retries": 2, "retry_backoff_s": 0.0},
            "validation": {"train_bars": 400, "test_bars": 200, "pbo_splits": 6},
        }
    )

import numpy as np
import pandas as pd
import pytest

from pairlab.engine.metrics import max_drawdown, sharpe
from pairlab.engine.reference import run_pair_loop
from pairlab.engine.vectorized import Thresholds, hedge_and_zscore, run_group, target_positions

PAIRS = [("C0M0", "C0M1"), ("C1M0", "C1M1"), ("N0", "N1")]
TH = Thresholds(
    entry=np.array([1.0, 1.5, 2.0, 2.5]),
    exit=np.array([0.0, 0.25, 0.5, 1.0]),
    stop=np.array([3.0, 4.0, 3.5, 99.0]),
)


def _yx(panel):
    return panel[[y for y, _ in PAIRS]], panel[[x for _, x in PAIRS]]


@pytest.mark.parametrize("delay,cost", [(0, 0.0), (1, 1.0), (3, 5.0)])
def test_vectorized_matches_reference_loop(panel, delay, cost):
    py, px = _yx(panel)
    res = run_group(py, px, 80, 15, TH, cost_bps=cost, delay=delay, max_cells=len(panel) * 3 * 2)
    for i, (y, x) in enumerate(PAIRS):
        for c in range(len(TH)):
            ret, held = run_pair_loop(panel[y], panel[x], 80, 15, TH.entry[c], TH.exit[c], TH.stop[c], cost, delay)
            assert res.pair_metrics["sharpe"][i, c] == pytest.approx(sharpe(ret[:, None], 252)[0], abs=1e-10)
            assert res.pair_metrics["n_trades"][i, c] == ((held != 0) & (held != np.r_[0, held[:-1]])).sum()
    np.testing.assert_allclose(
        res.portfolio_returns[:, 1],
        np.mean([run_pair_loop(panel[y], panel[x], 80, 15, 1.5, 0.25, 4.0, cost, delay)[0] for y, x in PAIRS], axis=0),
        atol=1e-12,
    )


def test_hysteresis_semantics():
    z = np.array([np.nan, 0.0, 2.1, 1.0, 0.4, -2.2, -5.0, -2.2, 0.1])[:, None]
    pos = target_positions(z, Thresholds(np.array([2.0]), np.array([0.5]), np.array([4.0])))[:, 0, 0]
    #             nan  0   enter-short hold exit enter-long  stop  re-enter exit
    assert pos.tolist() == [0, 0, -1, -1, 0, 1, 0, 1, 0]


def test_no_lookahead(panel):
    py, px = _yx(panel)
    base = run_group(py, px, 60, 20, TH).portfolio_returns
    cut = 1000
    shocked_y = py.copy()
    shocked_y.iloc[cut:] *= 1.5
    shocked = run_group(shocked_y, px, 60, 20, TH).portfolio_returns
    np.testing.assert_array_equal(base[:cut], shocked[:cut])
    assert not np.allclose(base[cut:], shocked[cut:])


def test_hedge_ratio_recovers_true_beta():
    rng = np.random.default_rng(0)
    lx = np.cumsum(rng.normal(0, 0.01, 2000)) + 4
    ly = 0.8 * lx + rng.normal(0, 0.002, 2000)
    idx = pd.date_range("2020", periods=2000, freq="B")
    beta, _ = hedge_and_zscore(pd.DataFrame({"a": ly}, idx), pd.DataFrame({"a": lx}, idx), 250, 20)
    assert np.nanmedian(beta) == pytest.approx(0.8, abs=0.02)


def test_chunking_does_not_change_results(panel):
    py, px = _yx(panel)
    a = run_group(py, px, 60, 20, TH, max_cells=10**9)
    b = run_group(py, px, 60, 20, TH, max_cells=1)
    np.testing.assert_allclose(a.portfolio_returns, b.portfolio_returns, rtol=0, atol=1e-15)
    for k in a.pair_metrics:
        np.testing.assert_allclose(a.pair_metrics[k], b.pair_metrics[k], rtol=1e-12)


def test_cointegrated_pairs_beat_noise_pairs(panel):
    py, px = _yx(panel)
    sr = run_group(py, px, 120, 20, TH).pair_metrics["sharpe"]
    assert sr[:2].mean() > sr[2].mean()


def test_max_drawdown():
    r = np.array([0.1, -0.5, 0.2])[:, None]
    assert max_drawdown(r)[0] == pytest.approx(-0.5)

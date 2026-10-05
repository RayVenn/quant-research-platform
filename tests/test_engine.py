import numpy as np
import pandas as pd
import pytest

from quantlab.engine.metrics import max_drawdown, sharpe
from quantlab.engine.portfolio import simulate
from quantlab.engine.reference import simulate_loop
from quantlab.strategies.pairs import PairsStrategy
from quantlab.strategies.pairs.engine import Thresholds, hedge_and_zscore, target_positions

PAIRS = [["C0M0", "C0M1"], ["C1M0", "C1M1"], ["N0", "N1"]]


@pytest.mark.parametrize("delay,cost,borrow", [(0, 0.0, 0.0), (1, 1.0, 0.0), (3, 5.0, 75.0)])
def test_simulator_matches_reference_loop(market, delay, cost, borrow):
    rng = np.random.default_rng(1)
    prices = market.close.to_numpy().copy()
    prices[100:130, 2] = np.nan  # halted / not yet listed
    W = rng.normal(0, 0.3, (3, *prices.shape)).transpose(1, 2, 0)
    W[::7] = np.nan
    rets = pd.DataFrame(prices).pct_change(fill_method=None).to_numpy()
    sim = simulate(W, rets, cost, delay, borrow)
    for c in range(W.shape[2]):
        ref = simulate_loop(W[:, :, c], prices, cost, delay, borrow)
        np.testing.assert_allclose(sim.returns[:, c], ref, atol=1e-12)


def test_hysteresis_semantics():
    z = np.array([np.nan, 0.0, 2.1, 1.0, 0.4, -2.2, -5.0, -2.2, 0.1])[:, None]
    pos = target_positions(z, Thresholds(np.array([2.0]), np.array([0.5]), np.array([4.0])))[:, 0, 0]
    #             nan  0   enter-short hold exit enter-long  stop  re-enter exit
    assert pos.tolist() == [0, 0, -1, -1, 0, 1, 0, 1, 0]


def test_hedge_ratio_recovers_true_beta():
    rng = np.random.default_rng(0)
    lx = np.cumsum(rng.normal(0, 0.01, 2000)) + 4
    ly = 0.8 * lx + rng.normal(0, 0.002, 2000)
    idx = pd.date_range("2020", periods=2000, freq="B")
    beta, _ = hedge_and_zscore(pd.DataFrame({"a": ly}, idx), pd.DataFrame({"a": lx}, idx), 250, 20)
    assert np.nanmedian(beta) == pytest.approx(0.8, abs=0.02)


def _pairs(market, pairs=PAIRS):
    s = PairsStrategy({"pairs": pairs})
    syms = sorted({x for p in pairs for x in p})
    m = market.select(syms)
    return PairsStrategy({"pairs": pairs}, s.fit(m)), m


def test_pairs_weights_are_dollar_neutral_and_hedged(market):
    s, m = _pairs(market)
    p = {"beta_window": 60, "z_window": 20, "entry_z": 1.5, "exit_z": 0.0, "stop_z": 4.0}
    W = s.positions_batch(m, [p])[:, :, 0]
    beta, _ = s.signals(m, 60, 20)
    col = {c: i for i, c in enumerate(m.symbols)}
    for i, (y, x) in enumerate(PAIRS):
        wy, wx = W[:, col[y]], W[:, col[x]]
        on = wy != 0
        assert on.any()
        np.testing.assert_allclose((np.abs(wy) + np.abs(wx))[on], 1 / len(PAIRS))  # gross 1 per pair
        np.testing.assert_allclose(wx[on], -wy[on] * beta[on, i])  # hedge ratio respected


def test_pairs_batch_matches_single_and_mixed_groups(market):
    s, m = _pairs(market)
    grid = [{"beta_window": bw, "z_window": 20, "entry_z": e, "exit_z": 0.0, "stop_z": 4.0}
            for bw in (60, 120) for e in (1.5, 2.0)]
    batch = s.positions_batch(m, grid)
    for c, p in enumerate(grid):
        np.testing.assert_array_equal(batch[:, :, c], s.positions(m, p))


def test_cointegrated_pairs_beat_noise_pairs(market):
    p = {"beta_window": 120, "z_window": 20, "entry_z": 1.5, "exit_z": 0.0, "stop_z": 4.0}
    srs = []
    for pairs in ([["C0M0", "C0M1"], ["C1M0", "C1M1"]], [["N0", "N1"]]):
        s, m = _pairs(market, pairs)
        srs.append(sharpe(simulate(s.positions_batch(m, [p]), m.returns.to_numpy()).returns, 252)[0])
    assert srs[0] > srs[1]


def test_max_drawdown():
    r = np.array([0.1, -0.5, 0.2])[:, None]
    assert max_drawdown(r)[0] == pytest.approx(-0.5)

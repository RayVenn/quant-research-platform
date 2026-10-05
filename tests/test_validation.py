import numpy as np
import pytest

from quantlab.validation.stats import deflated_sharpe, effective_trials, pbo_cscv, probabilistic_sharpe
from quantlab.validation.walkforward import make_folds


def test_folds_rolling_and_expanding():
    f = make_folds(1000, 300, 100, "rolling", warmup=50)
    assert [(x.train_start, x.train_end, x.test_end) for x in f[:2]] == [(50, 350, 450), (150, 450, 550)]
    assert f[-1].test_end <= 1000
    e = make_folds(1000, 300, 100, "expanding", warmup=50)
    assert all(x.train_start == 50 for x in e)
    with pytest.raises(ValueError):
        make_folds(100, 300, 100)


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (2000, 50))
    assert pbo_cscv(noise, 10)["pbo"] > 0.3
    edge = noise.copy()
    edge[:, 0] += 0.003
    assert pbo_cscv(edge, 10)["pbo"] < 0.05


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(2)
    ret = rng.normal(0.0008, 0.01, 1500)
    trials = rng.normal(0, 0.03, 500)
    assert probabilistic_sharpe(ret) > deflated_sharpe(ret, trials)
    assert deflated_sharpe(ret, trials, n_trials=2) > deflated_sharpe(ret, trials, n_trials=500)


def test_effective_trials():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(500, 1))
    assert effective_trials(np.repeat(base, 20, axis=1)) == pytest.approx(1.0)
    assert effective_trials(rng.normal(size=(500, 20))) == pytest.approx(20, rel=0.1)

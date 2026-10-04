import textwrap

import numpy as np
import pytest

from quantlab.strategies.momentum import CrossSectionalMomentum
from quantlab.strategies.trend import MovingAverageCrossover
from quantlab.strategy import loader
from quantlab.strategy.audit import audit
from quantlab.strategy.base import Strategy

GOOD = textwrap.dedent('''
    import numpy as np
    from quantlab.strategy import Strategy

    class Reversal(Strategy):
        name = "reversal"
        def positions(self, data, p):
            r = data.close.pct_change(p["lookback"])
            return -np.sign(r).fillna(0) / data.close.shape[1]
''')
PEEKING = GOOD.replace("data.close.pct_change(p[\"lookback\"])", "data.close.pct_change(p[\"lookback\"]).shift(-1)")


def _write(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(src)
    return str(p)


def test_builtins_resolve_and_load():
    assert {"pairs", "momentum", "ma_crossover"} <= set(loader.available())
    for name in ("pairs", "momentum", "ma_crossover"):
        assert issubclass(loader.load_class(loader.resolve(name)), Strategy)
    ref = loader.resolve("quantlab.strategies.trend:MovingAverageCrossover")
    assert loader.load_class(ref) is MovingAverageCrossover


def test_file_plugin_runs_from_shipped_source(tmp_path, market):
    path = _write(tmp_path, "rev.py", GOOD)
    ref = loader.resolve(path)
    assert ref.source == GOOD and ref.ref == "rev.py"
    (tmp_path / "rev.py").unlink()  # workers never need the file, only the shipped source
    s = loader.instantiate(loader.StrategyRef(**ref.to_dict()))
    assert s.positions_batch(market, [{"lookback": 5}]).shape == (len(market), len(market.symbols), 1)


def test_code_hash_tracks_edits(tmp_path):
    a = loader.resolve(_write(tmp_path, "a.py", GOOD)).code_hash
    b = loader.resolve(_write(tmp_path, "a.py", GOOD + "\n# tweak\n")).code_hash
    assert a != b


def test_ambiguous_file_needs_class_name(tmp_path):
    src = GOOD + "\nclass Other(Reversal):\n    pass\n"
    path = _write(tmp_path, "two.py", src)
    with pytest.raises(TypeError, match="exactly one"):
        loader.load_class(loader.resolve(path))
    assert loader.load_class(loader.resolve(f"{path}:Other")).__name__ == "Other"


def test_unknown_strategy():
    with pytest.raises(KeyError, match="unknown strategy"):
        loader.resolve("nope")


def test_audit_passes_builtins_and_catches_lookahead(tmp_path, market):
    assert audit(CrossSectionalMomentum({"min_assets": 3}), market,
                 {"lookback": 60, "skip": 5, "quantile": 0.25, "rebalance": 5}).ok
    assert audit(MovingAverageCrossover(), market, {"fast": 10, "slow": 50}).ok
    assert audit(loader.instantiate(loader.resolve(_write(tmp_path, "ok.py", GOOD))), market, {"lookback": 5}).ok
    bad = loader.instantiate(loader.resolve(_write(tmp_path, "bad.py", PEEKING)))
    rep = audit(bad, market, {"lookback": 5})
    assert not rep.ok and "lookahead" in rep.problems[0]


def test_momentum_long_short_book(market):
    W = CrossSectionalMomentum({"min_assets": 3}).positions(market, {"lookback": 60, "quantile": 0.25,
                                                                     "rebalance": 21})
    live = W.abs().sum(axis=1) > 0
    np.testing.assert_allclose(W[live].sum(axis=1), 0, atol=1e-12)  # dollar neutral
    np.testing.assert_allclose(W[live].abs().sum(axis=1), 1)  # gross 1
    changes = (W.diff().abs().sum(axis=1) > 0).to_numpy().nonzero()[0]
    assert all(i % 21 == 0 for i in changes)  # only trades on rebalance bars


def test_ma_crossover_long_only(market):
    W = MovingAverageCrossover().positions(market, {"fast": 10, "slow": 50})
    assert (W >= 0).all().all() and W.sum(axis=1).max() <= 1 + 1e-12
    assert MovingAverageCrossover().valid({"fast": 50, "slow": 10}) is False

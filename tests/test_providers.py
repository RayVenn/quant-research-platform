import os

import numpy as np
import pandas as pd
import pytest

from quantlab.data import providers
from quantlab.data.ingest import ingest
from quantlab.data.providers.yahoo import YahooProvider
from quantlab.data.store import PriceStore

IDX = pd.date_range("2024-01-01", periods=5, freq="B")


def test_csv_wide_and_long(tmp_path):
    wide = pd.DataFrame({"ts": IDX, "AAA": np.arange(5) + 10.0, "BBB": [1, 2, np.nan, 4, 5.0]})
    wide.to_csv(tmp_path / "w.csv", index=False)
    bars = providers.get_provider("csv", path=str(tmp_path / "w.csv")).fetch()
    assert set(bars) == {"AAA", "BBB"} and len(bars["BBB"]) == 4
    long = pd.DataFrame({"ts": list(IDX) * 2, "symbol": ["AAA"] * 5 + ["BBB"] * 5,
                         "close": np.arange(10.0) + 1, "volume": 100})
    long.to_csv(tmp_path / "l.csv", index=False)
    bars = providers.get_provider("csv", path=str(tmp_path / "l.csv"), format="long").fetch(["BBB"])
    assert list(bars) == ["BBB"] and list(bars["BBB"].columns) == ["close", "volume"]


def test_provider_lookup_by_module_path():
    cls = type(providers.get_provider("quantlab.data.providers.csv:CSVProvider", path="x"))
    assert cls.__name__ == "CSVProvider"
    with pytest.raises(KeyError):
        providers.get_provider("nope")


def _fake_download(batch, **kw):
    cols = pd.MultiIndex.from_product([batch, ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(np.arange(len(IDX) * len(cols), dtype=float).reshape(len(IDX), -1) + 1, IDX, cols)
    if "DEAD" in batch:
        df[("DEAD", "Close")] = np.nan
    return df


def test_yahoo_normalizes_and_ingest_is_incremental(tmp_path, monkeypatch):
    yf = pytest.importorskip("yfinance")
    calls = []
    monkeypatch.setattr(yf, "download", lambda batch, **kw: calls.append(list(batch)) or _fake_download(batch))
    store = str(tmp_path / "store")
    res = ingest("yahoo", ["AAA", "BBB", "DEAD"], store, start="2024-01-01")
    assert res == {"fetched": ["AAA", "BBB"], "skipped": [], "missing": ["DEAD"]}
    ps = PriceStore(store)
    assert list(ps.read("AAA").columns) == ["open", "high", "low", "close", "volume"]
    assert ps.meta("AAA")["provider"] == "yahoo"
    m = ps.load(["AAA", "BBB"])
    assert m["volume"].shape == (5, 2) and m.close.shape == (5, 2)
    res = ingest("yahoo", ["AAA", "CCC"], store)
    assert res["skipped"] == ["AAA"] and calls[-1] == ["CCC"]  # cached symbols are not re-downloaded


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("QUANTLAB_NETWORK_TESTS"), reason="set QUANTLAB_NETWORK_TESTS=1")
def test_yahoo_live():
    bars = YahooProvider().fetch(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01")
    assert set(bars) == {"AAPL", "MSFT"} and len(bars["AAPL"]) > 15

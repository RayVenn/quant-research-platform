"""Parquet-backed price store.

Layout (local path or any fsspec URI, e.g. ``s3://bucket/prices``)::

    <root>/<SYMBOL>.parquet      columns: ts (UTC timestamp), close (+ optional OHLCV)

The store is intentionally simple: one file per symbol, append by rewrite. It is
the contract every backtest worker reads from, so workers on any machine see
identical inputs as long as they point at the same root.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from functools import lru_cache

import fsspec
import pandas as pd


class PriceStore:
    def __init__(self, root: str):
        self.root = root.rstrip("/")
        self.fs, self._path = fsspec.core.url_to_fs(self.root)

    def _file(self, symbol: str) -> str:
        return f"{self._path}/{symbol}.parquet"

    def symbols(self) -> list[str]:
        if not self.fs.exists(self._path):
            return []
        return sorted(
            p.rsplit("/", 1)[-1].removesuffix(".parquet")
            for p in self.fs.ls(self._path, detail=False)
            if p.endswith(".parquet")
        )

    def write(self, symbol: str, bars: pd.DataFrame) -> None:
        """Write bars for *symbol*. ``bars`` must have a DatetimeIndex and a ``close`` column."""
        if "close" not in bars.columns:
            raise ValueError("bars must contain a 'close' column")
        df = bars.copy()
        df.index = pd.DatetimeIndex(df.index, name="ts")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.fs.makedirs(self._path, exist_ok=True)
        tmp = self._file(symbol) + ".tmp"
        with self.fs.open(tmp, "wb") as fh:
            df.reset_index().to_parquet(fh, index=False)
        self.fs.mv(tmp, self._file(symbol))

    def write_panel(self, panel: pd.DataFrame) -> None:
        """Write a wide close-price panel (columns = symbols)."""
        for sym in panel.columns:
            self.write(sym, panel[[sym]].rename(columns={sym: "close"}).dropna())

    def read(self, symbol: str) -> pd.DataFrame:
        with self.fs.open(self._file(symbol), "rb") as fh:
            df = pd.read_parquet(fh)
        return df.set_index("ts").sort_index()

    def load_panel(
        self,
        symbols: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Wide close-price panel aligned on the union index, forward-filled within gaps."""
        syms = list(symbols) if symbols is not None else self.symbols()
        if not syms:
            raise ValueError(f"no symbols found in price store {self.root!r}")
        cols = {s: self.read(s)["close"] for s in syms}
        panel = pd.DataFrame(cols).sort_index()
        if start is not None:
            panel = panel.loc[pd.Timestamp(start, tz="UTC") :]
        if end is not None:
            panel = panel.loc[: pd.Timestamp(end, tz="UTC")]
        return panel.ffill()


def fingerprint(panel: pd.DataFrame) -> str:
    """Content hash of a price panel — identifies the exact data a run used."""
    h = hashlib.sha256()
    h.update(",".join(map(str, panel.columns)).encode())
    h.update(pd.util.hash_pandas_object(panel, index=True).values.tobytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=4)
def cached_panel(root: str, symbols: tuple[str, ...], start: str | None, end: str | None) -> pd.DataFrame:
    """Research panel for *symbols*, restricted to bars where every symbol trades.

    Cached per process so a worker loads the data once across many tasks. The
    driver and every worker call this same function, so they see identical inputs.
    """
    return PriceStore(root).load_panel(symbols, start, end).dropna(how="any")

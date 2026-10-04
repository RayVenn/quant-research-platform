"""Parquet-backed market data store.

Layout (local path or any fsspec URI, e.g. ``s3://bucket/prices``)::

    <root>/<SYMBOL>.parquet      columns: ts (UTC), close [, open, high, low, volume]
    <root>/_meta/<SYMBOL>.json   provenance: provider, fetched_at, first/last bar

One file per symbol. Every backtest worker reads from this store, so workers on
any machine see identical inputs as long as they point at the same root.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

import fsspec
import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class MarketData:
    """Aligned wide panels (bars × symbols) handed to strategies.

    ``close`` is always present. ``open/high/low/volume`` are present when the
    provider supplied them. Prices are forward-filled across gaps; bars before a
    symbol's first trade stay NaN (the engine treats them as untradable).
    """

    close: pd.DataFrame
    fields: dict[str, pd.DataFrame] = field(default_factory=dict)
    periods_per_year: float = 252.0

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def __len__(self) -> int:
        return len(self.close)

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name == "close":
            return self.close
        if name not in self.fields:
            raise KeyError(f"field {name!r} not available (have: close, {', '.join(self.fields)})")
        return self.fields[name]

    @property
    def returns(self) -> pd.DataFrame:
        """Simple close-to-close returns; NaN where the symbol wasn't trading."""
        return self.close.pct_change(fill_method=None)

    def slice(self, start: int, stop: int) -> MarketData:
        return MarketData(
            self.close.iloc[start:stop], {k: v.iloc[start:stop] for k, v in self.fields.items()}, self.periods_per_year
        )

    def select(self, symbols: Iterable[str]) -> MarketData:
        syms = list(symbols)
        return MarketData(self.close[syms], {k: v[syms] for k, v in self.fields.items()}, self.periods_per_year)


class PriceStore:
    def __init__(self, root: str):
        self.root = str(root).rstrip("/")
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

    def has(self, symbol: str) -> bool:
        return self.fs.exists(self._file(symbol))

    def write(self, symbol: str, bars: pd.DataFrame, meta: dict | None = None) -> None:
        """Write bars for *symbol*: DatetimeIndex plus a ``close`` column (optional OHLCV)."""
        df = bars.rename(columns=str.lower)
        if "close" not in df.columns:
            raise ValueError("bars must contain a 'close' column")
        df = df[[c for c in FIELDS if c in df.columns]].astype("float64")
        df.index = pd.DatetimeIndex(df.index, name="ts")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["close"])
        self.fs.makedirs(self._path, exist_ok=True)
        tmp = self._file(symbol) + ".tmp"
        with self.fs.open(tmp, "wb") as fh:
            df.reset_index().to_parquet(fh, index=False)
        self.fs.mv(tmp, self._file(symbol))
        if meta is not None:
            self.fs.makedirs(f"{self._path}/_meta", exist_ok=True)
            info = {**meta, "first": str(df.index[0]) if len(df) else None,
                    "last": str(df.index[-1]) if len(df) else None, "bars": len(df)}
            with self.fs.open(f"{self._path}/_meta/{symbol}.json", "w") as fh:
                fh.write(json.dumps(info, default=str))

    def write_panel(self, panel: pd.DataFrame, meta: dict | None = None) -> None:
        """Write a wide close-price panel (columns = symbols)."""
        for sym in panel.columns:
            self.write(sym, panel[[sym]].rename(columns={sym: "close"}).dropna(), meta)

    def read(self, symbol: str) -> pd.DataFrame:
        with self.fs.open(self._file(symbol), "rb") as fh:
            df = pd.read_parquet(fh)
        return df.set_index("ts").sort_index()

    def meta(self, symbol: str) -> dict:
        p = f"{self._path}/_meta/{symbol}.json"
        if not self.fs.exists(p):
            return {}
        with self.fs.open(p, "r") as fh:
            return json.loads(fh.read())

    def load(
        self,
        symbols: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        periods_per_year: float = 252.0,
    ) -> MarketData:
        syms = list(symbols) if symbols is not None else self.symbols()
        if not syms:
            raise ValueError(f"no symbols found in price store {self.root!r}")
        missing = [s for s in syms if not self.has(s)]
        if missing:
            raise FileNotFoundError(f"symbols not in store {self.root!r}: {missing}")
        frames = {s: self.read(s) for s in syms}
        idx = sorted(set().union(*(f.index for f in frames.values())))
        index = pd.DatetimeIndex(idx, name="ts")
        lo = pd.Timestamp(start, tz="UTC") if start else None
        hi = pd.Timestamp(end, tz="UTC") if end else None
        index = index[(index >= lo) if lo is not None else slice(None)]
        index = index[(index <= hi) if hi is not None else slice(None)]
        panels = {}
        for f in FIELDS:
            if all(f in fr.columns for fr in frames.values()):
                panel = pd.DataFrame({s: fr[f] for s, fr in frames.items()}).reindex(index)
                panels[f] = panel if f == "volume" else panel.ffill()
        close = panels.pop("close")
        keep = close.notna().any(axis=1).to_numpy()
        return MarketData(close[keep], {k: v[keep] for k, v in panels.items()}, periods_per_year)

    def load_panel(self, symbols=None, start=None, end=None) -> pd.DataFrame:
        return self.load(symbols, start, end).close


def fingerprint(data: MarketData | pd.DataFrame) -> str:
    """Content hash of market data — identifies the exact inputs a run used."""
    panels = [data] if isinstance(data, pd.DataFrame) else [data.close, *data.fields.values()]
    h = hashlib.sha256()
    for p in panels:
        h.update(",".join(map(str, p.columns)).encode())
        h.update(pd.util.hash_pandas_object(p, index=True).values.tobytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=8)
def cached_market(
    root: str, symbols: tuple[str, ...], start: str | None, end: str | None, periods_per_year: float = 252.0
) -> MarketData:
    """Per-process cache so a worker loads data once across many tasks. Driver and workers
    call this same function, so they see identical inputs."""
    return PriceStore(root).load(symbols, start, end, periods_per_year)


def tradable_mask(data: MarketData) -> np.ndarray:
    return data.close.notna().to_numpy()

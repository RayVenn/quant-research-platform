from __future__ import annotations

import pandas as pd

from quantlab.data.providers.base import DataProvider


class CSVProvider(DataProvider):
    """Reads closes from a CSV: ``wide`` (ts,SYM1,SYM2,…) or ``long`` (ts,symbol,close[,open,…])."""

    name = "csv"

    def __init__(self, path: str, format: str = "wide", ts_col: str = "ts", **options):
        super().__init__(**options)
        self.path, self.format, self.ts_col = path, format, ts_col

    def fetch(self, symbols=None, start=None, end=None, interval="1d") -> dict[str, pd.DataFrame]:
        df = pd.read_csv(self.path, parse_dates=[self.ts_col])
        out = {}
        if self.format == "long":
            for sym, g in df.groupby("symbol"):
                out[str(sym)] = g.drop(columns="symbol").set_index(self.ts_col)
        else:
            wide = df.set_index(self.ts_col)
            out = {str(s): wide[[s]].rename(columns={s: "close"}).dropna() for s in wide.columns}
        if symbols:
            out = {s: v for s, v in out.items() if s in set(symbols)}
        if start or end:
            out = {s: v.loc[start:end] for s, v in out.items()}
        return out

"""Yahoo Finance via ``yfinance`` — free daily/intraday bars, no API key.

Yahoo's data is unofficial and for research/personal use; for production trading,
plug in a licensed vendor through the same ``DataProvider`` interface.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from quantlab.data.providers.base import DataProvider

log = logging.getLogger("quantlab.data")


class YahooProvider(DataProvider):
    name = "yahoo"

    def __init__(self, batch_size: int = 50, retries: int = 3, auto_adjust: bool = True, **options):
        super().__init__(**options)
        self.batch_size, self.retries, self.auto_adjust = batch_size, retries, auto_adjust

    def _download(self, batch: list[str], start, end, interval) -> pd.DataFrame:
        import yfinance as yf

        for attempt in range(1, self.retries + 1):
            try:
                return yf.download(
                    batch, start=start, end=end, interval=interval, auto_adjust=self.auto_adjust,
                    group_by="ticker", progress=False, threads=True,
                )
            except Exception as exc:  # noqa: BLE001 - network/vendor errors are retried
                if attempt == self.retries:
                    raise
                log.warning("yahoo download failed (%s), retry %d/%d", exc, attempt, self.retries)
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def fetch(self, symbols, start=None, end=None, interval="1d") -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), self.batch_size):
            batch = list(symbols[i : i + self.batch_size])
            raw = self._download(batch, start, end, interval)
            for sym in batch:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym]
                else:
                    df = raw
                df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_")).dropna(subset=["close"])
                if not df.empty:
                    out[sym] = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]]
        return out

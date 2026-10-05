"""Pull bars from a provider into the PriceStore (incremental by default)."""

from __future__ import annotations

import logging
import time

from quantlab.data.providers import get_provider
from quantlab.data.store import PriceStore

log = logging.getLogger("quantlab.data")


def ingest(
    provider: str,
    symbols: list[str],
    store: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    refresh: bool = False,
    **provider_options,
) -> dict[str, list[str]]:
    ps = PriceStore(store)
    todo = list(symbols) if refresh else [s for s in symbols if not ps.has(s)]
    skipped = [s for s in symbols if s not in todo]
    fetched: list[str] = []
    if todo:
        log.info("fetching %d symbols from %s", len(todo), provider)
        bars = get_provider(provider, **provider_options).fetch(todo, start, end, interval)
        meta = {"provider": provider, "interval": interval, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                        time.gmtime())}
        for sym, df in bars.items():
            ps.write(sym, df, meta)
            fetched.append(sym)
    missing = [s for s in todo if s not in fetched]
    if missing:
        log.warning("provider %s returned no data for: %s", provider, missing)
    return {"fetched": fetched, "skipped": skipped, "missing": missing}

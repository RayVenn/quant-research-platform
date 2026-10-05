"""Production hand-off: compute today's target weights from a registered artifact.

Live trading never touches research code paths or outputs. It asks the registry
for the ``production`` artifact and calls :func:`target_weights`. File plugins run
from the exact source stored in the artifact. For installed plugins, a code-hash
mismatch with the validated version is reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantlab.data.ingest import ingest
from quantlab.data.store import PriceStore, tradable_mask
from quantlab.strategy.loader import StrategyRef, instantiate, resolve


def target_weights(artifact: dict[str, Any], refresh: bool = False, store: str | None = None) -> dict[str, Any]:
    s, d = artifact["strategy"], artifact["data"]
    sref = StrategyRef(s["ref"], s["code_hash"], s.get("source"))
    warnings = []
    if sref.source is None:
        try:
            if resolve(sref.ref).code_hash != sref.code_hash:
                warnings.append("installed strategy code differs from the validated version")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not verify strategy code: {exc}")
    store = store or d["store"]
    if refresh:
        if not d.get("provider"):
            raise ValueError("artifact has no data provider; refresh the store yourself")
        ingest(d["provider"], d["symbols"], store, d.get("start"), None, d.get("interval", "1d"), refresh=True,
               **d.get("provider_options", {}))
    market = PriceStore(store).load(d["symbols"], d.get("start"), None, d["periods_per_year"])
    strategy = instantiate(sref, s["options"], s["state"])
    W = strategy.positions_batch(market, [artifact["params"]])[:, :, 0]
    w = np.where(tradable_mask(market), np.nan_to_num(W), 0.0)[-1]
    return {
        "name": artifact["name"],
        "version": artifact["version"],
        "asof": str(market.index[-1]),
        "params": artifact["params"],
        "weights": {sym: round(float(x), 6) for sym, x in zip(market.symbols, w, strict=True) if abs(x) > 1e-9},
        "gross": round(float(np.abs(w).sum()), 6),
        "net": round(float(w.sum()), 6),
        "execution": f"targets use bars through {market.index[-1].date()}; fill after {artifact['costs']['delay']} "
                     "bar(s) as in the backtest",
        "warnings": warnings,
    }

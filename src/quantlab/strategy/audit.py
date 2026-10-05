"""Static safety checks for any strategy plugin, run before every sweep."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantlab.data.store import MarketData
from quantlab.strategy.base import Params, Strategy


class StrategyAuditError(RuntimeError):
    pass


@dataclass
class AuditReport:
    problems: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def audit(strategy: Strategy, data: MarketData, params: Params, cuts=(0.5, 0.8)) -> AuditReport:
    """Lookahead check: weights before bar k must not change when bars ≥ k are removed."""
    rep = AuditReport()
    full = strategy.positions_batch(data, [params])
    if full.shape != (len(data), len(data.symbols), 1):
        rep.problems.append(f"positions_batch returned {full.shape}, expected {(len(data), len(data.symbols), 1)}")
        return rep
    w = full[:, :, 0]
    if not np.isfinite(np.nan_to_num(w, nan=0.0)).all():
        rep.problems.append("weights contain ±inf")
    for frac in cuts:
        k = int(len(data) * frac)
        part = strategy.positions_batch(data.slice(0, k), [params])[:, :, 0]
        diff = np.abs(np.nan_to_num(part) - np.nan_to_num(w[:k]))
        if (diff > 1e-9).any():
            first = int(np.argmax((diff > 1e-9).any(axis=1)))
            rep.problems.append(
                f"lookahead: weights at {data.index[first].date()} change when data after "
                f"{data.index[k - 1].date()} is removed (max diff {diff.max():.3g}); "
                "signals must use only past and current bars"
            )
            break
    gross = np.abs(np.nan_to_num(w)).sum(axis=1)
    rep.stats = {"max_gross": float(gross.max()), "avg_gross": float(gross.mean()),
                 "pct_invested": float((gross > 1e-12).mean())}
    return rep

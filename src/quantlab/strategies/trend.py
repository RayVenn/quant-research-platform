"""Time-series trend following via moving-average crossover (built-in plugin)."""

from __future__ import annotations

import numpy as np

from quantlab.strategy.base import Strategy


class MovingAverageCrossover(Strategy):
    name = "ma_crossover"
    group_by = ("slow",)
    default_options = {"long_short": False}

    def valid(self, p):
        return p["fast"] < p["slow"]

    def warmup(self, p):
        return int(p["slow"])

    def positions(self, data, p):
        c = data.close
        sig = np.sign(c.rolling(int(p["fast"])).mean() - c.rolling(int(p["slow"])).mean())
        if not self.options["long_short"]:
            sig = sig.clip(lower=0)
        n = c.notna().sum(axis=1).replace(0, np.nan)
        return sig.div(n, axis=0).fillna(0.0)

    def describe(self, data, p):
        w = self.positions(data, p).iloc[-1]
        return {"in_trend": sorted(w[w > 0].index.tolist()), "short": sorted(w[w < 0].index.tolist())}

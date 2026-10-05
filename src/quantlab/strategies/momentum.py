"""Cross-sectional equity momentum (built-in plugin).

Rank assets on trailing return over ``lookback`` bars, skipping the most recent
``skip`` bars (the classic 12-1 setup is lookback=252, skip=21). Long the top
``quantile``, optionally short the bottom ``quantile``, and rebalance every
``rebalance`` bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategy.base import Strategy


class CrossSectionalMomentum(Strategy):
    name = "momentum"
    group_by = ("lookback", "skip")
    default_options = {"long_short": True, "min_assets": 5}

    def warmup(self, p):
        return int(p["lookback"] + p.get("skip", 0))

    def valid(self, p):
        return 0 < p["quantile"] <= 0.5

    def positions(self, data, p):
        c = data.close
        skip = int(p.get("skip", 0))
        mom = c.shift(skip) / c.shift(skip + int(p["lookback"])) - 1.0
        ranks = mom.rank(axis=1, pct=True)
        q = float(p["quantile"])
        long_ = (ranks > 1 - q).astype(float)
        short = (ranks <= q).astype(float)
        w_long = long_.div(long_.sum(axis=1).replace(0, np.nan), axis=0)
        w_short = short.div(short.sum(axis=1).replace(0, np.nan), axis=0)
        w = 0.5 * w_long.fillna(0) - 0.5 * w_short.fillna(0) if self.options["long_short"] else w_long.fillna(0)
        w = w.where(mom.notna().sum(axis=1) >= self.options["min_assets"], 0.0)
        every = int(p.get("rebalance", 1))
        on = pd.Series(np.arange(len(w)) % every == 0, index=w.index)
        return w.where(on, np.nan).ffill().fillna(0.0)

    def describe(self, data, p):
        w = self.positions(data, p).iloc[-1]
        return {"longs": sorted(w[w > 0].index.tolist()), "shorts": sorted(w[w < 0].index.tolist())}

"""Low-volatility anomaly: an example user strategy plugin.

Long the least volatile names (inverse-vol weighted), rebalanced periodically.
Shows the minimal plugin shape: subclass Strategy, implement ``positions``,
return target weights. Referenced from a job as ``ref: strategies/low_vol.py``.
"""

import numpy as np

from quantlab.strategy import MarketData, Params, Strategy


class LowVolatility(Strategy):
    name = "low_vol"
    group_by = ("vol_window",)

    def valid(self, p: Params) -> bool:
        return 0 < p["top_frac"] <= 1

    def warmup(self, p: Params) -> int:
        return int(p["vol_window"])

    def positions(self, data: MarketData, p: Params):
        vol = data.returns.rolling(int(p["vol_window"])).std()
        keep = vol.rank(axis=1, pct=True) <= p["top_frac"]
        inv = (1.0 / vol).where(keep)
        w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)  # 0, not NaN, so dropped names are sold
        hold = np.arange(len(w)) % int(p["rebalance"]) != 0
        w[hold] = np.nan  # between rebalances, carry the last target forward
        return w.ffill().fillna(0.0)

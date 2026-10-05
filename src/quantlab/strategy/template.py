"""Scaffolding for ``quantlab strategy new``."""

STRATEGY = '''"""{title} — a quantlab strategy plugin.

Run it:      quantlab run jobs/{name}.yaml
Check it:    quantlab strategy check jobs/{name}.yaml   (lookahead audit + universe fit)
"""

import numpy as np
import pandas as pd

from quantlab.strategy import MarketData, Params, Strategy


class {cls}(Strategy):
    name = "{name}"
    # Params that share expensive precomputation (one sweep task per value). Optional.
    group_by = ("lookback",)
    # Fixed configuration; override under `strategy.options` in the job.
    default_options = {{"long_only": True}}

    def valid(self, p: Params) -> bool:
        return p["lookback"] > 1

    def positions(self, data: MarketData, p: Params) -> pd.DataFrame:
        """Target weights (bars × symbols) using only data up to each bar."""
        close = data.close
        z = (close - close.rolling(p["lookback"]).mean()) / close.rolling(p["lookback"]).std()
        signal = -np.sign(z).where(z.abs() > p["threshold"], 0.0)  # fade stretched moves
        if self.options["long_only"]:
            signal = signal.clip(lower=0)
        n = close.notna().sum(axis=1).replace(0, np.nan)
        return signal.div(n, axis=0).fillna(0.0)
'''

JOB = '''name: {name}
description: {title}
data:
  provider: yahoo            # free, no API key; missing symbols are fetched on first run
  store: data/yahoo
  symbols: [AAPL, MSFT, AMZN, GOOGL, META, JPM, XOM, JNJ, PG, KO, PEP, WMT, HD, V, MA, UNH]
  start: "2015-01-01"
strategy:
  ref: {path}
  options: {{long_only: true}}
  grid:
    lookback: [10, 20, 40]
    threshold: [1.0, 1.5, 2.0]
costs: {{cost_bps: 2.0, delay: 1}}
execution: {{backend: process}}
validation: {{train_bars: 756, test_bars: 126}}
'''

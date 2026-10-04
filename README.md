# quant-research-platform (`quantlab`)

A self-hosted alternative to a production backtesting platform. Researchers write a
strategy as a small Python plugin and a YAML job. The platform handles everything else:

- **Data.** Free market data from Yahoo Finance (no API key), CSV import, or your own
  provider plugin. Bars are cached as Parquet with provenance and fingerprinted, so runs
  are reproducible.
- **Vectorized backtesting.** Strategies emit target weights. One shared portfolio
  simulator applies execution delay, transaction costs and short-borrow costs, and is
  checked against a bar-by-bar reference.
- **Distributed parameter sweeps.** The same job runs in-process, on a process pool, or on
  a Ray cluster (docker-compose locally, KubeRay in the cloud).
- **Standardized execution, monitoring and recovery.** Deterministic run and task IDs, a
  durable task-state DB, retries with backoff, crash recovery, `resume`, `status`, a JSONL
  event stream and Prometheus metrics.
- **Model validation.** Walk-forward out-of-sample testing, Deflated Sharpe, Probability of
  Backtest Overfitting (CSCV), plus strategy-specific checks (e.g. cointegration stability
  for pairs). All of it feeds explicit promotion gates.
- **Research → production.** Versioned artifacts (code hash and source, params, data
  fingerprint, validation) move `research → staging → production`. `quantlab signal`
  turns the production artifact into today's target weights.

Equities are the first asset class. Strategies only see a `MarketData` panel and return
weights, so other asset classes need new data providers and cost models, not new strategy code.

See **[docs/DESIGN.md](docs/DESIGN.md)** for the architecture.

## Quick start

```bash
make setup        # uv installs Python + locked dependencies (~1 min)
make demo         # offline: synthetic data → pairs strategy → sweep → validate → register
make demo-yahoo   # real, free Yahoo data: momentum, trend, and a user plugin
make test
```

## Write a strategy

```bash
quantlab strategy new my_reversion      # creates strategies/my_reversion.py + jobs/my_reversion.yaml
quantlab strategy check jobs/my_reversion.yaml   # fetch data, fit, lookahead audit
quantlab run jobs/my_reversion.yaml
```

A strategy maps market data and one parameter set to target weights (bars × symbols):

```python
import numpy as np
from quantlab.strategy import Strategy

class LowVolatility(Strategy):
    group_by = ("vol_window",)            # optional: params that share expensive precompute

    def warmup(self, p):                  # optional: bars before the first valid signal
        return p["vol_window"]

    def positions(self, data, p):         # required
        vol = data.returns.rolling(p["vol_window"]).std()
        keep = vol.rank(axis=1, pct=True) <= p["top_frac"]
        inv = (1 / vol).where(keep)
        return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
```

The weight on bar `t` uses data through the close of `t`. The fill happens `costs.delay`
bars later, and P&L accrues from the bar after that.

Optional hooks:

| Hook | Purpose |
|------|---------|
| `fit(data)` | Run once on the leading `fit_fraction` of history (e.g. universe selection). The result is available as `self.state` and is stored in the artifact. |
| `valid(params)` | Drop invalid grid points. |
| `positions_batch(data, params_list)` | Vectorize across parameter combos (see the built-in pairs strategy). |
| `checks(data, params, windows)` | Add strategy-specific validation gates. |
| `describe(data, params)` | Snapshot (e.g. current holdings) stored with the artifact. |

Where `strategy.ref` can point:

| `strategy.ref` | Example |
|----------------|---------|
| Built-in | `pairs`, `momentum`, `ma_crossover` |
| Local file | `strategies/low_vol.py` or `strategies/file.py:ClassName` |
| Python module | `mypkg.signals:Carry` |
| Installed plugin | Any package declaring a `quantlab.strategies` entry point |

File plugins are shipped to workers as source, so Ray and process workers don't need the
file. The source hash is part of the run ID, so editing a strategy always produces a new run.
Every run starts with a **lookahead audit**: data is truncated, and the run fails if earlier
weights change.

## Data

```bash
quantlab data providers
quantlab data fetch --symbols AAPL MSFT SPY --store data/yahoo --start 2015-01-01
quantlab data fetch --symbols-file universe.txt --store data/yahoo --refresh
quantlab data import-csv prices.csv --store data/mine --format long   # ts,symbol,close[,open,high,low,volume]
quantlab data list --store data/yahoo
```

Jobs with `data.provider: yahoo` fetch missing symbols automatically on the first run. Later
runs read the Parquet cache. Yahoo data is unofficial and meant for research. For production,
add a licensed vendor by subclassing `quantlab.data.providers.base.DataProvider` and
registering it under the `quantlab.data_providers` entry point (or pass `module:Class`).

## Run, monitor, recover, promote

```bash
quantlab run jobs/momentum_yahoo.yaml --backend process --workers 8
quantlab status                                  # progress, retries, errors, recent events
quantlab resume <run_id> --retry-failed          # continue after a crash / failure
quantlab registry list
quantlab registry promote <name> <run_id> --stage staging
quantlab registry promote <name> <run_id> --stage production   # blocked if validation failed
quantlab signal <name>                           # today's target weights from the production artifact
quantlab signal <name> --refresh                 # pull fresh bars first
```

Run outputs live in `runs/<run_id>/`:

| File | Contents |
|------|----------|
| `spec.yaml`, `manifest.json` | Exact inputs: spec, data fingerprint, code hash, fit state |
| `strategy_source.py` | Shipped source, for file plugins |
| `events.jsonl`, `metrics.prom` | Monitoring |
| `aggregate/combo_metrics.parquet`, `aggregate/returns.parquet` | Per-combo metrics and daily returns |
| `validation.json`, `oos_equity.parquet` | Validation report and out-of-sample equity |

## Cluster

```bash
make cluster-up WORKERS=4      # Ray head + workers via docker-compose
make cluster-demo
kubectl apply -f deploy/k8s/rayjob.yaml   # ephemeral autoscaling KubeRay cluster per sweep
```

Fault injection for testing recovery: `QUANTLAB_CHAOS_FAIL_RATE=0.3 quantlab run …`.

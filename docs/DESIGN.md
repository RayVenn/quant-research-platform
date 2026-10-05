# quantlab — System Design

## 1. Goals

A research-to-production backtesting platform. Any strategy can run on it; equities are
supported first. A researcher supplies two things:

1. A **strategy plugin**: a Python class that maps market data and parameters to target
   portfolio weights.
2. A **job spec** (YAML): universe, data source, parameter grid, costs, validation and gates.

The platform provides everything else:

| Concern | Provided by |
|---|---|
| Market data | Provider plugins (Yahoo Finance free/no key, CSV, custom), a Parquet cache, fingerprints |
| Backtesting | Shared vectorized portfolio simulator (delay, costs, borrow), checked against a loop oracle |
| Sweeps | Deterministic task planner; local / process / Ray executors |
| Operations | Durable task state, retries, timeouts, crash recovery, resume, events, Prometheus metrics |
| Validation | Walk-forward OOS, Deflated Sharpe, PBO, strategy-specific checks, promotion gates |
| Hand-off | Versioned registry artifacts, `research → staging → production`, `uv run quantlab signal` |
| Environments | `uv` lockfile, devcontainer, Docker image, docker-compose Ray, KubeRay RayJob |

Non-goals for now: tick/order-book simulation, live order routing, a web UI.

## 2. Architecture

```
            job.yaml + strategy plugin
                       │
   ┌───────────────────▼───────────────────────────────── control plane ─────┐
   │ resolve: data (provider → Parquet cache) → fit (leading window) → grid  │
   │          → lookahead audit → run_id = H(spec, data, code, fit state)    │
   │ plan:    group_by params × combo chunks → tasks (content-addressed IDs) │
   │ state:   SQLite/Postgres (runs, tasks, attempts, leases, events)        │
   └───────────┬──────────────────────────────────────────────▲──────────────┘
               │ self-describing task payloads                │ status
   ┌───────────▼───────── data plane (stateless workers) ─────┴──────────────┐
   │ load MarketData (cached per process) → strategy.positions_batch         │
   │ → portfolio.simulate (delay, costs, borrow) → metrics + returns         │
   │ → atomic Parquet write                     local | process pool | Ray   │
   └───────────┬─────────────────────────────────────────────────────────────┘
               ▼
   aggregate → validate (walk-forward, DSR, PBO, strategy checks) → registry
                                                        │
                                   research → staging → production → signal
```

The two planes are separated on purpose. Workers are pure functions of their payload, so
any worker can execute any task. Only the control plane holds state.

## 3. The strategy contract

```python
class Strategy:
    group_by: tuple[str, ...] = ()        # params sharing expensive precompute (one task per value)
    default_options: dict = {}            # fixed config, overridable in job.strategy.options
    def fit(self, data) -> dict: ...      # once, on the leading fit_fraction of history → self.state
    def valid(self, params) -> bool: ...
    def warmup(self, params) -> int: ...
    def positions(self, data, params) -> DataFrame | ndarray      # REQUIRED: (bars × symbols) weights
    def positions_batch(self, data, params_list) -> ndarray       # (bars × symbols × combos)
    def checks(self, data, params, windows) -> list[dict]: ...    # extra validation gates
    def describe(self, data, params) -> dict: ...                 # artifact snapshot
```

**Why target weights.** The weights interface is the one boundary every systematic equity
strategy fits through. Pairs, momentum, trend, factor and ML signals all reduce to "how
much of each asset do I want to hold". Because the strategy never computes its own P&L,
every strategy gets identical, audited treatment of fills, costs, borrow and
data gaps. A strategy cannot quietly give itself a better fill.

**Timing convention.** The weight at bar `t` may use data through the close of `t`. It is
filled at bar `t + delay`, and P&L accrues from `t + delay + 1`. `simulate()` applies
this lag. Strategies never shift their own signals.

**Performance.** The default `positions_batch` loops `positions()` over combos. That is
fine for most research. Heavy strategies override it to share work across combos. The
built-in pairs strategy computes rolling hedge ratios and z-scores once per
`(beta_window, z_window)` and applies every entry/exit/stop combo in one `(T, P, C)`
numpy pass. That runs at roughly 2.7k full-history backtests/s/core, about 40× a
bar-by-bar loop. `group_by` tells the planner to put such shared-precompute combos in
the same task. The worker bounds memory by chunking combos to `T × N × C ≤ max_cells`,
and chunking does not change results (tested).

**Loading and provenance.** `strategy.ref` resolves, in order, to:

1. A built-in name.
2. An installed `quantlab.strategies` entry point.
3. A `module:Class` path.
4. A `path/to/file.py[:Class]` file.

The resolved `StrategyRef` carries a hash of the code. File plugins also carry their
source, which is shipped inside task payloads and executed in a namespaced module on the
worker. As a result:

- Ray and process workers never need the user's file system or a package install.
- The code hash is part of the run ID, so editing a strategy can never reuse stale results.
- The artifact stores the exact source that was validated. Production executes that
  source, not whatever is on disk later. For installed plugins, `signal` warns if the
  installed code hash differs from the validated one.

**Lookahead audit.** Before any sweep, the strategy is run on the full history and on
truncated prefixes (50%, 80%). Weights on the shared prefix must be identical. Any
`shift(-1)`, full-sample normalization, or centered window fails the run before compute
is spent. The audit also records gross-exposure stats, which catches leverage bugs. A
real example during development: a forward-filled weight table that kept stale names
showed gross 1.9 instead of 1.0. `uv run quantlab strategy check` runs the audit interactively.

**Security.** Plugins are arbitrary Python. That is appropriate for a trusted research
team. Workers run in containers with only the data and run-output credentials they
need. In a multi-tenant deployment, run each team's workers in their own namespace or
service account, and require registry promotion to go through review (see §8).

Built-ins, all implemented through the same public API:

| Name | Strategy | Notable hooks |
|---|---|---|
| `pairs` | Statistical-arbitrage pairs: rolling OLS hedge, z-score hysteresis | `fit` screens cointegrated pairs; `checks` adds cointegration stability; vectorized `positions_batch` |
| `momentum` | Cross-sectional momentum (e.g. 12-1), long/short quantiles, periodic rebalance | `group_by=(lookback, skip)` |
| `ma_crossover` | Time-series moving-average trend following | `valid` enforces fast < slow |

`strategies/low_vol.py` is a user-plugin example, run via `jobs/low_vol_yahoo.yaml`.

## 4. Data

`DataProvider.fetch(symbols, start, end, interval)` returns adjusted OHLCV per symbol.
`ingest()` writes bars to the `PriceStore`:

- One Parquet file per symbol, plus `_meta/<symbol>.json` with provider, interval and fetch time.
- Any fsspec URL works, e.g. `s3://bucket/prices`.
- Ingestion is incremental: cached symbols are skipped unless `--refresh` is set.

Jobs with `data.provider` fetch missing symbols automatically.

- **Yahoo Finance (`yfinance`).** Free, no key, split- and dividend-adjusted, batched with
  retries. It is good for research, but unofficial, rate-limited and survivorship-biased
  (today's tickers only). Stooq was evaluated and rejected: it now serves a browser
  verification page instead of CSV.
- **CSV.** Wide or long format, for vendor exports and offline work.
- **Custom.** Subclass `DataProvider` and register it under the `quantlab.data_providers`
  entry point. Use this for licensed vendors (Polygon, Tiingo, Databento, Norgate, CRSP,
  ...). Production should use a point-in-time, survivorship-free source.

`MarketData` holds `close` plus optional `open/high/low/volume` panels aligned on a union
index. Missing bars are NaN. The simulator only lets a strategy hold assets with a price
on that bar (`tradable_mask`), and it treats missing returns as zero.

**Reproducibility.** The run ID includes a SHA-256 fingerprint of the exact panels used.
A refresh that revises history produces a new run, not a silent change.

## 5. Job model and determinism

```
run_id = name + H(research spec, data fingerprint, strategy code hash, fit state, platform version)
task_id = H(run_id, combo ids in task)
```

Execution settings (backend, workers, retries, chunk size) are excluded from the run ID.
The same research on a laptop or on 200 Ray workers is the same run, and re-running a
finished job recomputes nothing. Each task is a pure function of its payload and writes
its outputs atomically (temp file + rename). Retries, duplicate execution after a lost
lease, and resumes are therefore all safe.

`resume <run_id>` refuses to continue if the inputs now resolve to a different run ID
(changed data, code or spec). Resuming would otherwise mix results.

## 6. Execution, monitoring and recovery

- **Executors.** `local` (debugging), `process` (one machine) and `ray` (cluster) share
  one `submit/wait` interface. Ray's own task retries are off, so every attempt is
  recorded in our state DB.
- **State DB.** Runs, tasks, attempts, errors, worker IDs and durations. It uses SQLite
  in WAL mode locally and Postgres in shared deployments.
- **Failure handling.**
  - Per-task timeouts.
  - Exponential-backoff retries up to `max_retries`, with the traceback recorded per attempt.
  - Tasks that exhaust their retries stop the run as `partial`.
  - `resume --retry-failed` gives exhausted tasks a fresh retry budget after a fix.
- **Crash recovery.** On restart, tasks left `running` are checked for outputs. Tasks with
  outputs are marked healed; tasks without outputs are requeued.
- **Fault injection.** Set `QUANTLAB_CHAOS_FAIL_RATE`. Tests prove that transient
  failures are retried and that exhausted, crashed and resumed runs converge to the same
  results.
- **Monitoring.**
  - `events.jsonl`: structured log per run.
  - `metrics.prom`: progress, failures and throughput, for the Prometheus textfile
    collector or a pushgateway.
  - `uv run quantlab status`: progress, ETA, failing tasks with errors, recent events.
  - The Ray dashboard, for clusters.

## 7. Validation: why a sweep winner is not a strategy

Validation runs on per-combo return series the sweep already produced, so it costs
almost nothing extra.

1. **Walk-forward OOS.** Rolling or expanding folds after warmup. In each fold, pick the
   best in-sample combo and record its next-fold returns. The concatenated OOS series is
   what a researcher would actually have experienced.
2. **Deflated Sharpe (Bailey & López de Prado).** Tests the OOS Sharpe against the
   expected maximum Sharpe of N trials. N is reduced to an *effective* trial count from
   the average correlation between combos, so a grid of near-duplicates is not
   over-penalized and a grid of diverse strategies is not under-penalized.
3. **PBO via CSCV.** The probability that the in-sample winner ranks below median out of
   sample, computed across combinatorial splits.
4. **Strategy checks.** Plugins can add gates. For example, pairs requires at least k
   pairs to stay cointegrated across the walk-forward windows.

Gates (`min_oos_sharpe`, `min_dsr`, `max_pbo`, `max_oos_drawdown`, plus strategy checks)
are declared in the job file, so changes to them go through code review.

Examples on free Yahoo data (2012–2026) show the gates doing their job:

- **Large-cap momentum** was rejected (OOS Sharpe 0.15).
- **ETF trend following** was rejected for overfitting: OOS Sharpe 0.52, but PBO 0.82.
- **The low-vol user plugin** was rejected for overfitting: OOS Sharpe 0.73, but PBO 0.71.

## 8. Research → production

`run_job` registers every run as an immutable artifact in `research`, whether it passed
or failed. The artifact contains:

- Strategy ref, code hash, and source (for file plugins).
- Options, fit state (e.g. selected pairs), and the chosen params.
- Costs and the data spec, including provider, symbols and fingerprint.
- Code version, validation report and a `describe()` snapshot.

Promotion goes `research → staging → production`, one stage at a time:

- Promotion is blocked when validation failed. `--force` overrides this and is
  written to an append-only audit log.
- Rollback is promoting the previous version.
- Staging is for paper trading. Production is what live systems read.

`uv run quantlab signal <name>` is the production hand-off. It:

1. Loads the production artifact.
2. Optionally refreshes bars from the artifact's provider.
3. Rebuilds the exact validated strategy (shipped source, options and fit state).
4. Outputs today's target weights, gross/net exposure, the as-of bar and any warnings.

Execution (OMS/EMS) consumes those targets. The platform never routes orders itself.

## 9. Provisioning: days → minutes

- **Laptop.** `uv sync --extra ray` installs a pinned Python and the locked dependencies
  in about a minute. A devcontainer is included.
- **Single box or CI.** `uv run pytest` and the demo job (`uv run quantlab data synth`,
  then `uv run quantlab run jobs/demo.yaml`) run fully offline on synthetic data.
- **Cluster.**
  - `docker compose up -d --build --scale ray-worker=n ray-head ray-worker` starts a Ray
    head and workers from the same image.
  - `deploy/k8s/rayjob.yaml` creates an ephemeral, autoscaling KubeRay cluster per sweep
    and tears it down afterwards.
- **Researcher experience.** Researchers write a plugin and a YAML file. They never
  configure infrastructure.

## 10. Production deployment (recommended)

| Layer | Choice |
|---|---|
| Market data | Licensed, point-in-time vendor via a provider plugin. Parquet in S3/GCS (same `PriceStore` code via fsspec) |
| Run state | Postgres (swap the `StateStore` connection) |
| Compute | KubeRay on spot/preemptible nodes. Preemption is just a retry |
| Orchestration | Airflow/Argo for nightly data refresh, re-validation of production strategies, and signal generation |
| Monitoring | Prometheus + Grafana dashboards on `metrics.prom`, Loki for `events.jsonl`, alerts on failed tasks and stale data |
| Registry | S3 with object lock, or a DB-backed service. Production promotion restricted to approvers (CODEOWNERS / IAM) |
| Plugins | Strategy repos packaged with `quantlab.strategies` entry points, built into images by CI. File plugins for exploration |

## 11. Extension roadmap

- **Other asset classes.** Add providers (futures with roll adjustment, FX, crypto) and
  per-asset cost and margin models in the simulator. The strategy API does not change.
- **Execution realism.** Volume-participation limits from `volume`, spread/impact cost
  models, open-price fills, intraday bars.
- **Portfolio layer.** Combine multiple registered strategies with risk targeting and constraints.
- **Universe.** Point-in-time index membership, corporate actions and delistings for
  survivorship-free research.
- **Operations.**
  - Live-vs-backtest drift monitoring on staging/production.
  - Result caching across overlapping grids.
  - A web UI over the state DB and registry.

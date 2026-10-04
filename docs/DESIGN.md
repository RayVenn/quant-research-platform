# pairlab — System Design

## 1. Goals

| Goal | What it means in practice |
|---|---|
| Large-scale vectorized backtesting | Score thousands of pair × parameter configurations over a full history in seconds per core, with no lookahead and with costs included. |
| Distributed parameter sweeps | Fan the same job out from a laptop to a cluster without changing research code. |
| Model validation | Every candidate faces out-of-sample, multiple-testing and stability evidence before anyone sees it as "good". |
| Standardized execution, monitoring, recovery | One job format, one state machine, one set of metrics and events. Any failure can be resumed. |
| Research → production without infra work | Researchers write a YAML spec. The platform produces a versioned artifact that live trading consumes. |
| Provisioning in minutes | `make setup` (uv + lockfile) locally. An immutable image and an ephemeral Ray cluster per sweep in the cloud. |

Non-goals for v1: tick-level simulation, an order-book fill model, and live order routing. The registry is the hand-off to the existing execution system.

## 2. Architecture

```
            ┌───────────────────────── control plane ──────────────────────────┐
 YAML job → │ pairlab CLI / API → planner → run state DB (runs, tasks, events)   │
            └───────┬───────────────────────────────────────────▲──────────────┘
                    │ tasks (idempotent, content-addressed)     │ status            
            ┌───────▼───────────────────── data plane ──────────┴──────────────┐
            │ executor: local | process pool | Ray cluster (autoscaling)        │
            │   worker: load panel (cached) → vectorized engine → atomic write   │
            └───────┬───────────────────────────────────────────────────────────┘
                    │ Parquet results
 object store ◄─────┘   prices/ (immutable, fingerprinted)   runs/<run_id>/…   registry/
                    │
            validation (walk-forward, DSR, PBO, coint stability) → gates
                    │ pass
            strategy registry: research → staging → production  ──►  live trading
```

Layers (`src/pairlab/`):

| Layer | Module | Responsibility |
|---|---|---|
| Data | `data/store.py`, `screening.py` | Parquet per symbol on local disk or any fsspec URI (S3/GCS). Content fingerprint. Correlation prefilter plus Engle-Granger screening. |
| Engine | `engine/vectorized.py`, `metrics.py` | Pure function: (prices, windows, threshold grid) → returns and metrics arrays. |
| Orchestration | `sweep/spec.py`, `planner.py`, `state.py`, `executors.py`, `runner.py`, `worker.py` | Spec → deterministic tasks → executor. Retries, timeouts, recovery, monitoring. |
| Validation | `validation/walkforward.py`, `stats.py`, `report.py` | Out-of-sample selection, overfitting statistics, gates. |
| Registry | `registry/store.py` | Versioned artifacts, stage promotion, audit log. |
| Interface | `cli.py`, `pipeline.py` | One entry point for humans and CI. |

## 3. Vectorized engine

A pairs strategy has two kinds of parameters:

- **Window parameters** (`beta_window`, `z_window`) change the spread itself. They are expensive.
- **Threshold parameters** (`entry`, `exit`, `stop`) only change how one z-score series is turned into positions. They are cheap.

The engine computes the rolling OLS hedge ratio and the z-score **once per window setting for all pairs**, a `(T, P)` array. It then broadcasts thresholds into a `(T, P, C)` state machine (C = threshold combos). So one task is one window setting × many thresholds × many pairs.

- **Hedge ratio:** rolling `cov(y,x)/var(x)` on log prices. The spread is the rolling OLS residual `(y-ȳ) - β(x-x̄)`.
- **Hysteresis without a Python loop:** `entry`, `exit` and `stop` produce sparse "events". The held position is the forward-filled last event, computed with `np.maximum.accumulate` over event indices. This is O(T·P·C) and fully vectorized.
- **No lookahead:** positions are shifted by `delay` (default 1 bar) before multiplying by next-bar spread returns. `tests/test_engine.py::test_no_lookahead` perturbs the future and asserts that the past is unchanged.
- **Costs:** `cost_bps × |Δposition| × gross leverage (1+|β|)`.
- **Memory bound:** the C axis is chunked so that `T·P·C ≤ max_cells`. Results are identical across chunk sizes (tested).
- **Correctness oracle:** `engine/reference.py` is a plain bar-by-bar loop. Tests assert that both implementations agree to 1e-12.

Measured on this VM: 50 pairs × 259 combos × 2,520 bars = 12,950 backtests in 4.7 s on **one core**, about 2.7k backtests/s/core. That is about 40× the reference loop. With N cores or Ray workers, throughput scales close to linearly because tasks share nothing.

**At larger scale** (thousands of symbols, intraday bars): keep the same function signature and swap the array backend. Options are numba for the state machine, or CuPy/JAX for GPU, behind `run_group`. Partition data by symbol and time so a worker only loads the pairs in its task.

## 4. Job model and determinism

A job is a declarative `JobSpec` (YAML, Pydantic-validated) with these sections: `data`, `universe` (explicit pairs or a screen), `grid`, `costs`, `validation`, `gates`, `execution`.

- `research_hash` = hash of every section **except `execution`**. Moving from 4 local cores to 200 Ray workers does not change the result identity.
- `run_id` = `name` + hash(research_hash, data fingerprint, pair universe). The same inputs always give the same run. Changed data gives a new run. `resume` refuses to continue a run if its inputs changed.
- Task ID = hash(run_id, window setting, threshold chunk). Tasks are pure functions of their payload, so a retry or a duplicate execution is harmless.

This gives reproducibility (every number can be rebuilt from an artifact) and caching (re-running a finished job costs nothing).

## 5. Execution, monitoring and recovery

**State machine** (SQLite in WAL mode locally; Postgres in a shared deployment):

```
pending ──► running ──► succeeded
              │  ▲
              ▼  │ retry with exponential backoff (max_retries)
            failed (attempts exhausted) ──► `resume --retry-failed` resets
```

- **Executors** share one interface: `submit(fn, payload) → future`. `LocalExecutor` is for debugging. `ProcessExecutor` uses spawn, which avoids fork/BLAS issues. `RayExecutor` uses remote tasks with Ray's own retries turned off, so every retry is visible in *our* state DB instead of being hidden.
- **Atomic outputs:** workers write `metrics.parquet` and `returns.parquet` to a temp name and then rename. An output file is either complete or absent.
- **Timeouts:** a task that runs longer than `task_timeout_s` is marked failed and retried.
- **Crash recovery** (`recover()`, which runs on every `run` and `resume`): a `running` task whose output exists is marked succeeded. A `running` task with no output was orphaned by a dead driver or worker and goes back to `pending`. The driver can therefore be killed at any point (tested in `test_sweep.py::test_crash_recovery`).
- **Monitoring:**
  - `events.jsonl`: structured per-task events (start, success, failure with traceback, retry) for log shipping.
  - `metrics.prom`: Prometheus textfile metrics (tasks by state, backtests/s, ETA) for node-exporter/Grafana.
  - `pairlab status`: progress, failures and recent errors.
  - Ray dashboard on port 8265 for cluster-level CPU and memory.
- **Fault injection:** `PAIRLAB_CHAOS_FAIL_RATE` fails a deterministic fraction of attempts. Tests use it to prove that retries and recovery work.

## 6. Validation (why a sweep winner is not a strategy)

A grid of hundreds of configs will always contain one with a great in-sample Sharpe. Validation is a separate stage that runs on the already-computed return matrix, so it adds almost no compute:

1. **Walk-forward** (rolling or expanding folds). In each fold, choose the best config on train and record its test returns. The stitched OOS series is the strategy's honest track record, and it also shows parameter stability across folds.
2. **Deflated Sharpe Ratio** (Bailey & López de Prado). This is the probability that the true Sharpe exceeds the expected maximum Sharpe of N trials under the null. N is the **effective** number of trials, corrected for correlation between configs, so near-duplicate grid points are not overcounted.
3. **PBO via CSCV.** Split time into S blocks and, for every combination of half the blocks as in-sample, check where the IS winner ranks out of sample. PBO = P(the IS winner is below the OOS median).
4. **Cointegration stability.** Re-test each pair's Engle-Granger p-value in every fold window. Pairs that are not cointegrated in most windows get flagged.
5. **Gates** (`gates:` in the job) check minimum OOS Sharpe, minimum DSR, maximum PBO, maximum OOS drawdown and minimum stable pairs. Gates are code-reviewed config, not opinions.

## 7. Research → production

The registry (`registry/store.py`; files or object storage) stores one artifact per run. Each artifact holds:
- the chosen parameters
- the pair list and hedge-ratio settings
- the data fingerprint
- research_hash
- code version (git SHA baked into the image)
- the full validation report
- the stage

- Stages are `research → staging → production`, one step at a time.
- Promotion is blocked if the validation report failed. `--force` exists but is written to `history.jsonl` with the user's note.
- **Staging** = paper trading or shadow mode in the live system using the exact artifact. Move to production after the live-vs-backtest tracking error is within tolerance.
- The live trading service only ever reads `registry show NAME --stage production`. It never reads research outputs directly. Rollback means promoting the previous version.

## 8. Provisioning: days → minutes

| Layer | Mechanism | Time |
|---|---|---|
| Developer env | `uv.lock` + `make setup` (uv fetches Python and wheels), or `.devcontainer` | ~1 min |
| Image | `Dockerfile` builds a locked env with the git SHA baked in. CI builds it on every merge. | cached |
| Local cluster | `docker-compose.yml` (Ray head + scalable workers + CLI) | `make cluster-up WORKERS=8` |
| Cloud cluster | `deploy/k8s/rayjob.yaml` (KubeRay `RayJob`): an ephemeral autoscaling cluster (0→64 workers) that shuts down after the job | minutes, pay per sweep |
| CI | `.github/workflows/ci.yml`: lint, tests, demo run | per PR |

Researchers never touch servers. They run `pairlab run job.yaml --backend ray --ray-address …`, or submit the RayJob with their spec.

## 9. Production deployment (recommended)

| Concern | Local / v1 (in this repo) | Production |
|---|---|---|
| Market data | Parquet directory | S3/GCS Parquet (partitioned by symbol/date), same `PriceStore` via fsspec. Ingestion job writes immutable snapshots. |
| Run state | SQLite | Postgres (same schema). Lease and heartbeat columns for multiple drivers. |
| Results / registry | local dirs | Object store with versioning plus a registry table. Optionally MLflow as the UI. |
| Compute | process pool / compose | KubeRay on EKS/GKE with spot nodes (tasks are idempotent, so preemption is just a retry). |
| Scheduling | CLI | Airflow/Argo/Prefect triggers nightly re-validation of production strategies on new data. |
| Monitoring | events.jsonl, metrics.prom | Prometheus + Grafana + Loki. Alert on failure rate, stuck runs and validation drift. |
| Access | local | SSO, per-team namespaces and quotas, and RBAC so that only approvers can run `promote --stage production`. |

## 10. Scaling and extension roadmap

1. GPU or numba backend for `run_group`, and intraday bars with sessions and calendars.
2. Kalman-filter hedge ratios and multi-leg baskets: same task model, new engine function.
3. Shared run-state service (Postgres) plus a web UI over runs, the registry and validation reports.
4. Live drift monitor: compare production PnL with a backtest replayed on the same bars, and auto-demote when it breaches.
5. Fill and borrow model: short-availability and borrow-cost data per symbol.

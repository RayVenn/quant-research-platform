# pairs-research-platform (`pairlab`)

Research-to-production platform for **pairs-trading** strategies:

- **Vectorized backtesting.** One numpy pass scores every pair × every entry/exit/stop combo for a window setting. That is roughly 2.7k full-history backtests/s per core, about 40× a bar-by-bar loop.
- **Distributed parameter sweeps.** The same job runs in-process, on a multi-core process pool, or on a Ray cluster (docker-compose locally, KubeRay in the cloud).
- **Standardized execution, monitoring and recovery.** Declarative YAML job specs and deterministic run/task IDs. A durable task state DB tracks retries with backoff, crash recovery, `resume`, `status`, a JSONL event stream and Prometheus metrics.
- **Model validation.** Walk-forward out-of-sample testing, Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV) and cointegration-stability checks feed explicit promotion gates.
- **Strategy registry.** Versioned, reproducible artifacts (params, pairs, data fingerprint, code version) move through `research → staging → production` with an audit log. Live trading reads only `production`.

See **[docs/DESIGN.md](docs/DESIGN.md)** for the architecture.

## Quick start (≈2 minutes on a fresh machine)

```bash
make setup     # uv installs Python + locked dependencies
make demo      # synthetic data → screen → sweep → validate → register
make test
```

```text
run demo_pairs-…  status=validated  tasks={'succeeded': 27, 'failed': 0, ...}
check             value    rule      passed
oos_sharpe        2.77     >= 0.5    True
deflated_sharpe   0.976    >= 0.9    True
pbo               0.016    <= 0.5    True
...
```

## Everyday commands

```bash
pairlab data synth --store data/prices                     # or: data import-csv prices.csv --store data/prices
pairlab screen jobs/demo.yaml                              # cointegrated pair candidates
pairlab run jobs/demo.yaml --backend process --workers 8   # sweep + validate + register
pairlab status                                             # progress, retries, errors, recent events
pairlab resume <run_id> --retry-failed                     # continue after a crash / failure
pairlab registry list
pairlab registry promote demo_pairs <run_id> --stage staging
pairlab registry promote demo_pairs <run_id> --stage production
pairlab registry show demo_pairs --stage production        # what live trading consumes
```

Re-running the same job is free: the run ID comes from the spec plus the data fingerprint, and finished tasks are never recomputed.

## Cluster

```bash
make cluster-up WORKERS=4   # Ray head + 4 workers in docker-compose; dashboard on :8265
make cluster-demo
```

For Kubernetes, `deploy/k8s/rayjob.yaml` (KubeRay) brings up an autoscaling ephemeral cluster per sweep.

## Fault injection

`PAIRLAB_CHAOS_FAIL_RATE=0.3 pairlab run jobs/demo.yaml` makes about 30% of task attempts fail, which exercises retries and recovery.

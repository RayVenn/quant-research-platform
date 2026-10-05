"""Declarative job specification: the one standard way to describe a research run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSpec(_Model):
    store: str = Field(description="PriceStore root (local path or fsspec URI such as s3://bucket/prices)")
    provider: str | None = Field(None, description="Fetch missing symbols from this provider (e.g. 'yahoo')")
    provider_options: dict[str, Any] = {}
    symbols: list[str] | None = Field(None, description="Universe; default = every symbol in the store")
    start: str | None = None
    end: str | None = None
    interval: str = "1d"
    periods_per_year: float = 252.0


class StrategySpec(_Model):
    ref: str = Field(description="built-in name | installed plugin | module:Class | path/to/file.py[:Class]")
    options: dict[str, Any] = Field({}, description="Fixed (not swept) strategy configuration")
    grid: dict[str, list[Any]] = Field(description="Swept parameters: name -> candidate values")
    fit_fraction: float = Field(0.4, gt=0, le=1, description="Leading share of history passed to Strategy.fit")

    @model_validator(mode="after")
    def _nonempty(self) -> StrategySpec:
        empty = [k for k, v in self.grid.items() if not v]
        if not self.grid or empty:
            raise ValueError(f"strategy.grid needs at least one value per parameter (empty: {empty})")
        return self


class CostSpec(_Model):
    cost_bps: float = Field(1.0, description="Commission + slippage per unit of traded notional")
    delay: int = Field(1, ge=0, description="Bars between signal and fill")
    borrow_bps: float = Field(0.0, description="Annual borrow cost on short notional")


class ExecutionSpec(_Model):
    backend: Literal["local", "process", "ray"] = "process"
    max_workers: int | None = None
    combos_per_task: int = 200
    max_cells: int = Field(20_000_000, description="Bound on T×N×C cells per engine pass (memory)")
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    task_timeout_s: float | None = None
    ray_address: str | None = None


class ValidationSpec(_Model):
    scheme: Literal["rolling", "expanding"] = "rolling"
    train_bars: int = 504
    test_bars: int = 126
    warmup_bars: int | None = Field(None, description="Bars skipped before the first fold (default: Strategy.warmup)")
    pbo_splits: int = 10


class GateSpec(_Model):
    min_oos_sharpe: float = 0.5
    min_dsr: float = 0.90
    max_pbo: float = 0.5
    max_oos_drawdown: float = -0.25


class JobSpec(_Model):
    name: str = Field(pattern=r"^[a-zA-Z0-9_\-]+$")
    description: str = ""
    data: DataSpec
    strategy: StrategySpec
    costs: CostSpec = CostSpec()
    execution: ExecutionSpec = ExecutionSpec()
    validation: ValidationSpec = ValidationSpec()
    gates: GateSpec = GateSpec()

    @classmethod
    def from_yaml(cls, path: str | Path) -> JobSpec:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)

    def research_hash(self) -> str:
        """Hash of everything that affects results. Execution settings are excluded so a run
        can be resumed on a different backend or worker count without invalidating results."""
        payload = self.model_dump(mode="json", exclude={"execution", "description"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

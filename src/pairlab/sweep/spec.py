"""Declarative job specification — the single, standardized way to describe a research run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSpec(_Model):
    store: str = Field(description="PriceStore root (local path or fsspec URI such as s3://bucket/prices)")
    symbols: list[str] | None = None
    start: str | None = None
    end: str | None = None
    periods_per_year: float = 252.0


class ScreenSpec(_Model):
    min_corr: float = 0.7
    max_pvalue: float = 0.05
    max_half_life: float = 60.0
    max_pairs: int = 20
    train_fraction: float = Field(0.4, gt=0, le=1, description="Screen only on the leading part of history")


class UniverseSpec(_Model):
    pairs: list[tuple[str, str]] | None = None
    screen: ScreenSpec | None = None

    @model_validator(mode="after")
    def _one_source(self) -> UniverseSpec:
        if not self.pairs and self.screen is None:
            self.screen = ScreenSpec()
        return self


class GridSpec(_Model):
    beta_window: list[int]
    z_window: list[int]
    entry_z: list[float]
    exit_z: list[float]
    stop_z: list[float]


class CostSpec(_Model):
    cost_bps: float = 1.0
    delay: int = Field(1, ge=0, description="Bars between signal and fill")


class ExecutionSpec(_Model):
    backend: Literal["local", "process", "ray"] = "process"
    max_workers: int | None = None
    combos_per_task: int = 200
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    task_timeout_s: float | None = None
    ray_address: str | None = None


class ValidationSpec(_Model):
    scheme: Literal["rolling", "expanding"] = "rolling"
    train_bars: int = 504
    test_bars: int = 126
    warmup_bars: int | None = Field(None, description="Bars skipped before the first fold (default: max window)")
    pbo_splits: int = 10
    coint_pvalue: float = 0.10
    min_coint_stability: float = 0.5


class GateSpec(_Model):
    min_oos_sharpe: float = 0.5
    min_dsr: float = 0.90
    max_pbo: float = 0.5
    max_oos_drawdown: float = -0.25
    min_pairs: int = 2


class JobSpec(_Model):
    name: str = Field(pattern=r"^[a-zA-Z0-9_\-]+$")
    description: str = ""
    data: DataSpec
    universe: UniverseSpec = UniverseSpec()
    grid: GridSpec
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

"""File-based strategy registry: the hand-off point between research and production.

Layout::

    <root>/<strategy>/<version>/artifact.json
    <root>/<strategy>/stages.json        {"staging": "<version>", "production": "<version>"}
    <root>/<strategy>/history.jsonl      append-only audit log of promotions

Stages are ordered research → staging → production; a version can only move
one stage forward at a time, and only if its validation gates passed (unless
forced, which is recorded in the audit log). Live trading reads the
``production`` artifact and nothing else.
"""

from __future__ import annotations

import getpass
import json
import time
from pathlib import Path
from typing import Any

STAGES = ("research", "staging", "production")


class RegistryError(RuntimeError):
    pass


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        return self.root / name

    def _stages(self, name: str) -> dict[str, str]:
        p = self._dir(name) / "stages.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _audit(self, name: str, **rec: Any) -> None:
        rec = {"ts": time.time(), "actor": getpass.getuser(), **rec}
        with open(self._dir(name) / "history.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def register(self, artifact: dict[str, Any]) -> Path:
        name, version = artifact["name"], artifact["version"]
        d = self._dir(name) / version
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "artifact.json.tmp"
        tmp.write_text(json.dumps(artifact, indent=2, default=str))
        tmp.replace(d / "artifact.json")
        self._audit(name, action="register", version=version, passed=artifact["validation"]["passed"])
        return d / "artifact.json"

    def get(self, name: str, version: str | None = None, stage: str | None = None) -> dict[str, Any]:
        if version is None:
            if stage is None:
                raise RegistryError("pass a version or a stage")
            version = self._stages(name).get(stage)
            if version is None:
                raise RegistryError(f"no version of {name!r} in stage {stage!r}")
        p = self._dir(name) / version / "artifact.json"
        if not p.exists():
            raise RegistryError(f"{name}:{version} not found")
        return json.loads(p.read_text())

    def stage_of(self, name: str, version: str) -> str:
        stages = self._stages(name)
        for s in reversed(STAGES[1:]):
            if stages.get(s) == version:
                return s
        return "research"

    def promote(self, name: str, version: str, stage: str, force: bool = False, note: str = "") -> None:
        if stage not in STAGES[1:]:
            raise RegistryError(f"stage must be one of {STAGES[1:]}")
        art = self.get(name, version)
        current = self.stage_of(name, version)
        if not force:
            if not art["validation"]["passed"]:
                raise RegistryError(f"{name}:{version} failed validation gates; use --force to override")
            if STAGES.index(stage) != STAGES.index(current) + 1:
                raise RegistryError(f"{name}:{version} is in {current!r}; can only promote to the next stage")
        stages = self._stages(name)
        previous = stages.get(stage)
        stages[stage] = version
        tmp = self._dir(name) / "stages.json.tmp"
        tmp.write_text(json.dumps(stages, indent=2))
        tmp.replace(self._dir(name) / "stages.json")
        self._audit(name, action="promote", version=version, stage=stage, previous=previous, forced=force, note=note)

    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for sdir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for vdir in sorted(p for p in sdir.iterdir() if (p / "artifact.json").exists()):
                art = json.loads((vdir / "artifact.json").read_text())
                out.append(
                    {
                        "name": sdir.name,
                        "version": vdir.name,
                        "stage": self.stage_of(sdir.name, vdir.name),
                        "passed": art["validation"]["passed"],
                        "oos_sharpe": art["validation"]["oos"]["sharpe"],
                        "created_at": art["created_at"],
                    }
                )
        return out

    def history(self, name: str) -> list[dict[str, Any]]:
        p = self._dir(name) / "history.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []

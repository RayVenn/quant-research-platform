"""Pluggable execution backends with a common submit / wait / result interface.

* ``local``   — in-process, sequential (debugging, tests)
* ``process`` — multi-core on one host via a process pool
* ``ray``     — a Ray cluster (laptop, docker-compose, or KubeRay on Kubernetes)
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any

Fn = Callable[[dict[str, Any]], dict[str, Any]]


class Executor:
    name = "base"
    capacity = 1

    def submit(self, fn: Fn, payload: dict[str, Any]) -> Any: ...
    def wait(self, handles: list[Any], timeout: float) -> list[Any]: ...
    def result(self, handle: Any) -> dict[str, Any]: ...
    def cancel(self, handle: Any) -> None: ...
    def is_broken(self) -> bool:
        return False

    def shutdown(self) -> None: ...


class _Done:
    def __init__(self, value: Any = None, error: BaseException | None = None):
        self.value, self.error = value, error


class LocalExecutor(Executor):
    name = "local"

    def submit(self, fn, payload):
        try:
            return _Done(value=fn(payload))
        except Exception as exc:  # noqa: BLE001 - surfaced to the runner
            return _Done(error=exc)

    def wait(self, handles, timeout):
        return list(handles)

    def result(self, handle):
        if handle.error is not None:
            raise handle.error
        return handle.value

    def cancel(self, handle):
        pass


class ProcessExecutor(Executor):
    name = "process"

    def __init__(self, max_workers: int | None = None):
        self.capacity = max_workers or os.cpu_count() or 2
        self._pool = cf.ProcessPoolExecutor(max_workers=self.capacity, mp_context=mp.get_context("spawn"))

    def submit(self, fn, payload):
        return self._pool.submit(fn, payload)

    def wait(self, handles, timeout):
        done, _ = cf.wait(handles, timeout=timeout, return_when=cf.FIRST_COMPLETED)
        return list(done)

    def result(self, handle):
        return handle.result()

    def cancel(self, handle):
        handle.cancel()

    def is_broken(self) -> bool:
        return bool(getattr(self._pool, "_broken", False))

    def shutdown(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


class RayExecutor(Executor):
    name = "ray"

    def __init__(self, address: str | None = None, max_workers: int | None = None):
        try:
            import ray
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Ray backend requires the 'ray' extra: uv sync --extra ray") from exc
        self._ray = ray
        if not ray.is_initialized():
            ray.init(address=address or os.environ.get("RAY_ADDRESS"), ignore_reinit_error=True,
                     log_to_driver=False)
        self.capacity = max_workers or int(ray.cluster_resources().get("CPU", 1))
        self._remote_cache: dict[Fn, Any] = {}

    def submit(self, fn, payload):
        remote = self._remote_cache.get(fn)
        if remote is None:
            # Retries are owned by the runner so they are visible in the state store.
            remote = self._remote_cache[fn] = self._ray.remote(max_retries=0)(fn)
        return remote.remote(payload)

    def wait(self, handles, timeout):
        done, _ = self._ray.wait(handles, num_returns=1, timeout=timeout)
        return done

    def result(self, handle):
        return self._ray.get(handle)

    def cancel(self, handle):
        self._ray.cancel(handle, force=True)


def make_executor(backend: str, max_workers: int | None = None, ray_address: str | None = None) -> Executor:
    if backend == "local":
        return LocalExecutor()
    if backend == "process":
        return ProcessExecutor(max_workers)
    if backend == "ray":
        return RayExecutor(ray_address, max_workers)
    raise ValueError(f"unknown backend {backend!r}")

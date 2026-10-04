"""Durable run/task state (SQLite). The driver is the only writer.

Task lifecycle::

    pending ──submit──▶ running ──ok──▶ succeeded
       ▲                  │
       └──retry (attempts < max)──┤
                          └──exhausted──▶ failed
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    manifest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    n_combos INTEGER NOT NULL,
    worker TEXT,
    submitted_at REAL,
    finished_at REAL,
    duration_s REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS tasks_run_status ON tasks(run_id, status);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,
    run_id TEXT NOT NULL,
    level TEXT NOT NULL,
    task_id TEXT,
    message TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # runs -------------------------------------------------------------
    def upsert_run(self, run_id: str, name: str, manifest: dict[str, Any]) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT INTO runs(run_id, name, status, created_at, updated_at, manifest) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(run_id) DO UPDATE SET updated_at=excluded.updated_at, manifest=excluded.manifest",
            (run_id, name, "created", now, now, json.dumps(manifest)),
        )

    def set_run_status(self, run_id: str, status: str) -> None:
        self.conn.execute("UPDATE runs SET status=?, updated_at=? WHERE run_id=?", (status, time.time(), run_id))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # tasks ------------------------------------------------------------
    def register_tasks(self, tasks: list[dict[str, Any]]) -> int:
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO tasks(task_id, run_id, status, payload, n_combos) VALUES (?,?,?,?,?)",
            [(t["task_id"], t["run_id"], "pending", json.dumps(t), len(t["combo_ids"])) for t in tasks],
        )
        return cur.rowcount

    def tasks(self, run_id: str, status: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM tasks WHERE run_id=?", [run_id]
        if status:
            q += " AND status=?"
            args.append(status)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY task_id", args)]

    def mark_running(self, task_id: str) -> int:
        self.conn.execute(
            "UPDATE tasks SET status='running', attempts=attempts+1, submitted_at=?, error=NULL WHERE task_id=?",
            (time.time(), task_id),
        )
        return self.conn.execute("SELECT attempts FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]

    def mark_succeeded(self, task_id: str, worker: str | None, duration_s: float | None) -> None:
        self.conn.execute(
            "UPDATE tasks SET status='succeeded', worker=?, finished_at=?, duration_s=?, error=NULL WHERE task_id=?",
            (worker, time.time(), duration_s, task_id),
        )

    def mark_failed(self, task_id: str, error: str, max_attempts: int) -> str:
        """Record a failed attempt; returns the new status (pending if it will be retried)."""
        attempts = self.conn.execute("SELECT attempts FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
        status = "pending" if attempts < max_attempts else "failed"
        self.conn.execute(
            "UPDATE tasks SET status=?, finished_at=?, error=? WHERE task_id=?",
            (status, time.time(), error[-4000:], task_id),
        )
        return status

    def reset(self, run_id: str, from_status: str, to_status: str = "pending", reset_attempts: bool = False) -> int:
        extra = ", attempts=0" if reset_attempts else ""
        cur = self.conn.execute(
            f"UPDATE tasks SET status=?{extra} WHERE run_id=? AND status=?", (to_status, run_id, from_status)
        )
        return cur.rowcount

    def counts(self, run_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks WHERE run_id=? GROUP BY status", (run_id,)
        ).fetchall()
        out = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    # events -----------------------------------------------------------
    def log_event(self, run_id: str, level: str, message: str, task_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, run_id, level, task_id, message) VALUES (?,?,?,?,?)",
            (time.time(), run_id, level, task_id, message),
        )

    def events(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY ts DESC LIMIT ?", (run_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

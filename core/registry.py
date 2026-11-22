"""SQLite experiment registry.

All writes are deterministic and transactional: each public mutating method runs
inside a single transaction, so an exception (including an LLM/API failure
raised mid-operation) rolls back cleanly and never leaves partial state.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

EXPERIMENT_ID_RE = re.compile(r"^EXP-\d{3,}$")

STATUSES = ("planned", "running", "completed", "failed", "analyzed", "archived", "cancelled")

# Allowed status transitions (deterministic state machine).
TRANSITIONS: dict[str, set[str]] = {
    "planned": {"running", "cancelled"},
    "running": {"completed", "failed"},
    "completed": {"analyzed", "failed"},
    "failed": {"planned", "archived"},
    "analyzed": {"archived"},
    "archived": set(),
    "cancelled": set(),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    hypothesis   TEXT NOT NULL DEFAULT '',
    model_type   TEXT NOT NULL DEFAULT '',
    params       TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL,
    is_baseline  INTEGER NOT NULL DEFAULT 0,
    parent_id    TEXT,
    result_path  TEXT,
    predictions_path TEXT,
    notes        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    name          TEXT NOT NULL,
    value         REAL NOT NULL,
    source        TEXT NOT NULL,
    recorded_at   TEXT NOT NULL,
    PRIMARY KEY (experiment_id, name)
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    kind          TEXT NOT NULL,
    detail        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


class RegistryError(Exception):
    pass


class InvalidTransition(RegistryError):
    pass


class UnknownExperiment(RegistryError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Experiment:
    id: str
    name: str
    hypothesis: str
    model_type: str
    params: dict[str, Any]
    status: str
    is_baseline: bool
    parent_id: str | None
    result_path: str | None
    predictions_path: str | None
    notes: str
    created_at: str
    updated_at: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "hypothesis": self.hypothesis,
            "model_type": self.model_type, "params": self.params, "status": self.status,
            "is_baseline": self.is_baseline, "parent_id": self.parent_id,
            "result_path": self.result_path, "predictions_path": self.predictions_path,
            "notes": self.notes, "metrics": dict(self.metrics),
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


class Registry:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()  # Streamlit sessions share one connection
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic unit of work. Nested calls join the outer transaction."""
        with self._lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ reads
    def exists(self, exp_id: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        return row is not None

    def get(self, exp_id: str) -> Experiment:
        row = self._conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if row is None:
            raise UnknownExperiment(f"Experiment {exp_id} is not registered")
        return self._row_to_exp(row)

    def list(self, status: str | None = None) -> list[Experiment]:
        if status:
            rows = self._conn.execute("SELECT * FROM experiments WHERE status = ? ORDER BY id", (status,))
        else:
            rows = self._conn.execute("SELECT * FROM experiments ORDER BY id")
        return [self._row_to_exp(r) for r in rows.fetchall()]

    def baseline(self) -> Experiment | None:
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE is_baseline = 1 ORDER BY id LIMIT 1"
        ).fetchone()
        return self._row_to_exp(row) if row else None

    def metrics(self, exp_id: str) -> dict[str, float]:
        rows = self._conn.execute(
            "SELECT name, value FROM metrics WHERE experiment_id = ? ORDER BY name", (exp_id,)
        ).fetchall()
        return {r["name"]: float(r["value"]) for r in rows}

    def metric_sources(self, exp_id: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT name, source FROM metrics WHERE experiment_id = ?", (exp_id,)
        ).fetchall()
        return {r["name"]: r["source"] for r in rows}

    def events(self, exp_id: str | None = None) -> list[dict[str, Any]]:
        if exp_id:
            rows = self._conn.execute("SELECT * FROM events WHERE experiment_id = ? ORDER BY id", (exp_id,))
        else:
            rows = self._conn.execute("SELECT * FROM events ORDER BY id")
        return [dict(r) for r in rows.fetchall()]

    def next_id(self) -> str:
        nums = [int(r["id"].split("-")[1]) for r in self._conn.execute("SELECT id FROM experiments")]
        return f"EXP-{(max(nums) + 1 if nums else 1):03d}"

    def snapshot(self) -> list[tuple]:
        """Full deterministic dump; used to prove failures don't corrupt the registry."""
        out: list[tuple] = []
        for table in ("experiments", "metrics", "events"):
            out.extend((table, *tuple(r)) for r in self._conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2"))
        return out

    # ----------------------------------------------------------------- writes
    def create_experiment(
        self,
        name: str,
        *,
        exp_id: str | None = None,
        hypothesis: str = "",
        model_type: str = "",
        params: dict[str, Any] | None = None,
        status: str = "planned",
        is_baseline: bool = False,
        parent_id: str | None = None,
        notes: str = "",
    ) -> Experiment:
        if status not in STATUSES:
            raise RegistryError(f"Unknown status {status!r}")
        with self.transaction() as c:
            exp_id = exp_id or self.next_id()
            if not EXPERIMENT_ID_RE.match(exp_id):
                raise RegistryError(f"Invalid experiment id {exp_id!r}")
            if self.exists(exp_id):
                raise RegistryError(f"Experiment {exp_id} already exists")
            if parent_id and not self.exists(parent_id):
                raise UnknownExperiment(f"Parent experiment {parent_id} is not registered")
            ts = _now()
            c.execute(
                "INSERT INTO experiments (id, name, hypothesis, model_type, params, status, is_baseline,"
                " parent_id, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (exp_id, name, hypothesis, model_type, json.dumps(params or {}, sort_keys=True),
                 status, int(is_baseline), parent_id, notes, ts, ts),
            )
            self._event(c, exp_id, "created", f"status={status}")
        return self.get(exp_id)

    def transition(self, exp_id: str, new_status: str, note: str = "") -> Experiment:
        with self.transaction() as c:
            exp = self.get(exp_id)
            if new_status not in TRANSITIONS.get(exp.status, set()):
                raise InvalidTransition(f"{exp_id}: cannot move from '{exp.status}' to '{new_status}'")
            c.execute("UPDATE experiments SET status = ?, updated_at = ? WHERE id = ?", (new_status, _now(), exp_id))
            self._event(c, exp_id, "status", f"{exp.status} -> {new_status}" + (f" ({note})" if note else ""))
        return self.get(exp_id)

    def set_artifacts(self, exp_id: str, *, result_path: str | None = None, predictions_path: str | None = None) -> None:
        with self.transaction() as c:
            self.get(exp_id)
            if result_path is not None:
                c.execute("UPDATE experiments SET result_path = ?, updated_at = ? WHERE id = ?", (result_path, _now(), exp_id))
            if predictions_path is not None:
                c.execute("UPDATE experiments SET predictions_path = ?, updated_at = ? WHERE id = ?", (predictions_path, _now(), exp_id))
            self._event(c, exp_id, "artifacts", f"result={result_path} predictions={predictions_path}")

    def record_metrics(self, exp_id: str, values: dict[str, float], source: str) -> None:
        if source not in {"file", "computed"}:
            raise RegistryError("Metric source must be 'file' or 'computed'")
        with self.transaction() as c:
            self.get(exp_id)
            for name, value in values.items():
                v = float(value)
                if not math.isfinite(v):
                    raise RegistryError(f"Metric {name} for {exp_id} is not finite")
                c.execute(
                    "INSERT INTO metrics (experiment_id, name, value, source, recorded_at) VALUES (?,?,?,?,?)"
                    " ON CONFLICT(experiment_id, name) DO UPDATE SET value=excluded.value,"
                    " source=excluded.source, recorded_at=excluded.recorded_at",
                    (exp_id, name, v, source, _now()),
                )
            self._event(c, exp_id, "metrics", f"{source}: " + ", ".join(f"{k}={float(v):.4f}" for k, v in values.items()))

    def add_note(self, exp_id: str, note: str) -> None:
        with self.transaction() as c:
            exp = self.get(exp_id)
            notes = (exp.notes + "\n" if exp.notes else "") + note
            c.execute("UPDATE experiments SET notes = ?, updated_at = ? WHERE id = ?", (notes, _now(), exp_id))
            self._event(c, exp_id, "note", note[:200])

    # --------------------------------------------------------------- helpers
    def _event(self, c: sqlite3.Connection, exp_id: str, kind: str, detail: str) -> None:
        c.execute("INSERT INTO events (experiment_id, kind, detail, created_at) VALUES (?,?,?,?)",
                  (exp_id, kind, detail, _now()))

    def _row_to_exp(self, row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"], name=row["name"], hypothesis=row["hypothesis"], model_type=row["model_type"],
            params=json.loads(row["params"] or "{}"), status=row["status"],
            is_baseline=bool(row["is_baseline"]), parent_id=row["parent_id"],
            result_path=row["result_path"], predictions_path=row["predictions_path"],
            notes=row["notes"], created_at=row["created_at"], updated_at=row["updated_at"],
            metrics=self.metrics(row["id"]),
        )

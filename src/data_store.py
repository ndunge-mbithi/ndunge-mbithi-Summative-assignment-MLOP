"""Durable storage for uploads, retraining jobs, predictions and request metrics.

SQLite in WAL mode on the persistent volume. It is enough here: one host, a handful
of API replicas sharing the volume, and write volumes in the hundreds per second.
Moving to PostgreSQL would only require swapping this module (all SQL is plain).

High-frequency writes (request metrics, prediction logs) go through ``AsyncWriter``,
a per-process background thread that batches inserts, so a slow disk or a lock held
by another replica never adds latency to a prediction.
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS upload_batches (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, label TEXT, n_received INTEGER, n_accepted INTEGER,
    n_rejected INTEGER, n_duplicate INTEGER, errors_json TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sha256 TEXT UNIQUE NOT NULL, label TEXT NOT NULL,
    batch_id TEXT, original_name TEXT, raw_path TEXT NOT NULL, array_path TEXT, width INTEGER, height INTEGER,
    stats_json TEXT, source TEXT NOT NULL DEFAULT 'upload', status TEXT NOT NULL,
    used_in_version TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, progress REAL DEFAULT 0,
    params_json TEXT, epochs_json TEXT DEFAULT '[]', log_json TEXT DEFAULT '[]', result_json TEXT,
    parent_version TEXT, candidate_version TEXT, error TEXT, pid INTEGER,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, heartbeat_at TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, ts REAL NOT NULL, model_version TEXT, predicted_class TEXT,
    p_parasitized REAL, confidence REAL, low_confidence INTEGER, latency_ms REAL, stats_json TEXT,
    replica TEXT, true_label TEXT, feedback_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
CREATE TABLE IF NOT EXISTS requests (
    ts REAL NOT NULL, replica TEXT, method TEXT, path TEXT, status INTEGER, ms REAL
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.conn().executescript(SCHEMA)  # executescript manages its own transaction

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=15, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=15000")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self, immediate: bool = False):
        c = self.conn()
        c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def q(self, sql: str, args=()) -> list[dict]:
        return [dict(r) for r in self.conn().execute(sql, args).fetchall()]

    def one(self, sql: str, args=()) -> dict | None:
        r = self.conn().execute(sql, args).fetchone()
        return dict(r) if r else None

    # --------------------------------------------------------------- uploads
    def create_batch(self, batch_id: str, label: str | None, note: str = "") -> None:
        with self.tx() as c:
            c.execute("INSERT INTO upload_batches(id, created_at, label, n_received, n_accepted, n_rejected, "
                      "n_duplicate, errors_json, note) VALUES (?,?,?,0,0,0,0,'[]',?)", (batch_id, utcnow(), label, note))

    def finish_batch(self, batch_id: str, received: int, accepted: int, rejected: int, duplicate: int,
                     errors: list[dict]) -> None:
        with self.tx() as c:
            c.execute("UPDATE upload_batches SET n_received=?, n_accepted=?, n_rejected=?, n_duplicate=?, errors_json=? "
                      "WHERE id=?", (received, accepted, rejected, duplicate, json.dumps(errors[:200]), batch_id))

    def sample_exists(self, sha: str) -> bool:
        return self.one("SELECT 1 AS x FROM samples WHERE sha256=?", (sha,)) is not None

    def add_sample(self, **row) -> int | None:
        row.setdefault("created_at", utcnow())
        cols = ",".join(row)
        try:
            with self.tx() as c:
                cur = c.execute(f"INSERT INTO samples({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
                return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # duplicate sha256

    def update_sample(self, sample_id: int, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE samples SET {sets} WHERE id=?", (*fields.values(), sample_id))

    def trainable_samples(self) -> list[dict]:
        return self.q("SELECT * FROM samples WHERE status='preprocessed' AND used_in_version IS NULL ORDER BY id")

    def mark_samples_used(self, ids: list[int], version: str) -> None:
        with self.tx() as c:
            c.executemany("UPDATE samples SET used_in_version=?, status='used' WHERE id=?", [(version, i) for i in ids])

    def dataset_summary(self) -> dict:
        rows = self.q("SELECT label, status, COUNT(*) n FROM samples GROUP BY label, status")
        batches = self.q("SELECT * FROM upload_batches ORDER BY created_at DESC LIMIT 50")
        for b in batches:
            b["errors"] = json.loads(b.pop("errors_json") or "[]")
        total = sum(r["n"] for r in rows)
        pending = sum(r["n"] for r in rows if r["status"] == "preprocessed")
        return {"total_samples": total, "pending_for_training": pending, "by_label_status": rows, "batches": batches}

    # ------------------------------------------------------------------ jobs
    def create_job_exclusive(self, kind: str, params: dict, parent: str | None) -> tuple[str | None, dict | None]:
        """Create a job unless one is already queued/running (atomic across replicas)."""
        with self.tx(immediate=True) as c:
            busy = c.execute("SELECT * FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
            if busy:
                return None, dict(busy)
            job_id = new_id("job")
            c.execute("INSERT INTO jobs(id, kind, status, stage, params_json, parent_version, created_at, heartbeat_at) "
                      "VALUES (?,?,?,?,?,?,?,?)", (job_id, kind, "queued", "queued", json.dumps(params), parent,
                                                    utcnow(), utcnow()))
            return job_id, None

    def update_job(self, job_id: str, **fields) -> None:
        fields["heartbeat_at"] = utcnow()
        for k in ("result_json", "params_json", "epochs_json", "log_json"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k])
        sets = ",".join(f"{k}=?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE jobs SET {sets} WHERE id=?", (*fields.values(), job_id))

    def append_job_log(self, job_id: str, message: str, key: str = "log_json", entry: dict | None = None) -> None:
        with self.tx(immediate=True) as c:
            row = c.execute(f"SELECT {key} FROM jobs WHERE id=?", (job_id,)).fetchone()
            items = json.loads(row[0] or "[]") if row else []
            items.append(entry if entry is not None else {"at": utcnow(), "message": message})
            c.execute(f"UPDATE jobs SET {key}=?, heartbeat_at=? WHERE id=?", (json.dumps(items), utcnow(), job_id))

    def get_job(self, job_id: str) -> dict | None:
        j = self.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._decode_job(j) if j else None

    def list_jobs(self, limit: int = 20) -> list[dict]:
        return [self._decode_job(j) for j in self.q("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))]

    @staticmethod
    def _decode_job(j: dict) -> dict:
        for k in ("params_json", "epochs_json", "log_json", "result_json"):
            j[k.removesuffix("_json")] = json.loads(j.pop(k) or "null")
        return j

    def fail_stale_jobs(self, max_silence_s: int = 600) -> int:
        """Jobs whose worker stopped heart-beating (e.g. container restart) are marked failed."""
        cutoff = datetime.fromtimestamp(time.time() - max_silence_s, timezone.utc).isoformat(timespec="seconds")
        with self.tx() as c:
            cur = c.execute("UPDATE jobs SET status='failed', error='worker stopped responding (stale heartbeat)', "
                            "finished_at=? WHERE status IN ('queued','running') AND heartbeat_at < ?", (utcnow(), cutoff))
            return cur.rowcount

    # ----------------------------------------------------------- predictions
    def add_feedback(self, pred_id: str, true_label: str) -> bool:
        with self.tx() as c:
            cur = c.execute("UPDATE predictions SET true_label=?, feedback_at=? WHERE id=?", (true_label, utcnow(), pred_id))
            return cur.rowcount == 1

    def get_prediction(self, pred_id: str) -> dict | None:
        return self.one("SELECT * FROM predictions WHERE id=?", (pred_id,))


class AsyncWriter:
    """Batches INSERTs on a daemon thread. ``put`` never blocks the request path."""

    def __init__(self, store: Store, flush_every: float = 0.5, max_queue: int = 50_000):
        self.store = store
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.flush_every = flush_every
        self.dropped = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self._t.start()

    def put(self, table: str, row: dict) -> None:
        try:
            self.q.put_nowait((table, row))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.flush_every)
            self.flush()

    def flush(self) -> None:
        items = []
        while True:
            try:
                items.append(self.q.get_nowait())
            except queue.Empty:
                break
        if not items:
            return
        grouped: dict[tuple, list] = {}
        for table, row in items:
            grouped.setdefault((table, tuple(row)), []).append(tuple(row.values()))
        try:
            with self.store.tx() as c:
                for (table, cols), values in grouped.items():
                    c.executemany(f"INSERT OR IGNORE INTO {table}({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                                  values)
        except sqlite3.Error:
            log.exception("async writer flush failed; %d rows lost", len(items))

    def close(self) -> None:
        self._stop.set()
        self.flush()

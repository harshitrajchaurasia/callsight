import sqlite3
import time

from .config import DB_PATH
from .models import CallResult


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, client_id TEXT, created REAL DEFAULT (unixepoch()), "
                 "result TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (call_id TEXT, at REAL, text TEXT, vec BLOB)")  # see ask.py
    conn.execute("CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY, created REAL DEFAULT (unixepoch()), "
                 "owner TEXT, action TEXT NOT NULL)")  # see command.py
    conn.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts REAL NOT NULL, call_id TEXT, user TEXT, "
                 "kind TEXT NOT NULL, text TEXT NOT NULL)")  # what happened, in words people read in the app; see activity.py
    return conn


def save(conn, r: CallResult) -> None:
    # an upsert, not INSERT OR REPLACE: re-saving an edited call keeps its place (and created time) in the list
    conn.execute("INSERT INTO calls (id, client_id, result) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                 "client_id = excluded.client_id, result = excluded.result", (r.id, r.client_id, r.model_dump_json()))
    conn.commit()


def _load(result: str, created: float) -> CallResult:
    return CallResult.model_validate_json(result).model_copy(update={"created": created})


def get(conn, call_id: str) -> CallResult | None:
    row = conn.execute("SELECT result, created FROM calls WHERE id = ?", (call_id,)).fetchone()
    return _load(*row) if row else None


def all_calls(conn) -> list[CallResult]:
    return [_load(*r) for r in conn.execute("SELECT result, created FROM calls ORDER BY created DESC, rowid DESC")]


def delete(conn, call_id: str) -> None:
    conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
    conn.execute("DELETE FROM chunks WHERE call_id = ?", (call_id,))
    conn.commit()


def event(conn, kind: str, text: str, *, call_id: str | None = None, user: str | None = None) -> None:
    conn.execute("INSERT INTO events (ts, call_id, user, kind, text) VALUES (?, ?, ?, ?, ?)",
                 (time.time(), call_id, user, kind, text))
    conn.commit()


def events(conn, *, call_id: str | None = None, limit: int = 300) -> list[dict]:
    """Newest first. With call_id, that call's history; without, everything (the caller filters by who may see it)."""
    q, args = ("WHERE call_id = ?", (call_id,)) if call_id else ("", ())
    rows = conn.execute(f"SELECT id, ts, call_id, user, kind, text FROM events {q} ORDER BY id DESC LIMIT ?", (*args, limit))
    return [dict(zip(("id", "ts", "call_id", "user", "kind", "text"), row)) for row in rows]

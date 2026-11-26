import sqlite3

from .config import DB_PATH
from .models import CallResult


def connect(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, client_id TEXT, created REAL DEFAULT (unixepoch()), "
                 "result TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (call_id TEXT, at REAL, text TEXT, vec BLOB)")  # see ask.py
    return conn


def save(conn, r: CallResult) -> None:
    conn.execute("INSERT OR REPLACE INTO calls (id, client_id, result) VALUES (?, ?, ?)", (r.id, r.client_id, r.model_dump_json()))
    conn.commit()


def get(conn, call_id: str) -> CallResult | None:
    row = conn.execute("SELECT result FROM calls WHERE id = ?", (call_id,)).fetchone()
    return CallResult.model_validate_json(row[0]) if row else None


def all_calls(conn) -> list[CallResult]:
    return [CallResult.model_validate_json(r[0]) for r in conn.execute("SELECT result FROM calls ORDER BY created DESC")]

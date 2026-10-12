from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from models.task import Task


def _db_path() -> str:
    return os.getenv("DB_PATH", "./research.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                status TEXT NOT NULL,
                assigned_agent TEXT DEFAULT 'researcher',
                evidence TEXT DEFAULT '[]',
                evidence_texts TEXT DEFAULT '[]',
                unverified_claims TEXT DEFAULT '[]',
                result TEXT,
                retries INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 2,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(goal_id) REFERENCES goals(id))"""
        )
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that older databases lack (CREATE TABLE IF NOT EXISTS
    won't touch them)."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    for name, ddl in (
        ("unverified_claims", "TEXT DEFAULT '[]'"),
        ("tokens_in", "INTEGER DEFAULT 0"),
        ("tokens_out", "INTEGER DEFAULT 0"),
        ("claims", "TEXT DEFAULT '[]'"),
        ("source_titles", "TEXT DEFAULT '{}'"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {ddl}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- goals ----

def create_goal(query: str) -> dict:
    gid = uuid4().hex[:8]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO goals (id, query, created_at) VALUES (?,?,?)",
            (gid, query, _now()),
        )
    return {"id": gid, "query": query}


def get_goal(goal_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
    return dict(row) if row else None


# ---- tasks ----

def create_task(goal_id: str, task: Task) -> None:
    d = task.model_dump(mode="json")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO tasks
               (id, goal_id, title, description, status, assigned_agent,
                evidence, evidence_texts, unverified_claims, result, retries,
                max_retries, tokens_in, tokens_out, claims, source_titles,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d["id"], goal_id, d["title"], d["description"], d["status"],
                d["assigned_agent"], json.dumps(d["evidence"]),
                json.dumps(d["evidence_texts"]),
                json.dumps(d["unverified_claims"]), d["result"], d["retries"],
                d["max_retries"], d["tokens_in"], d["tokens_out"],
                json.dumps(d["claims"]), json.dumps(d["source_titles"]),
                d["created_at"], d["updated_at"],
            ),
        )


def _row_to_task_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["evidence"] = json.loads(d["evidence"] or "[]")
    d["evidence_texts"] = json.loads(d["evidence_texts"] or "[]")
    d["unverified_claims"] = json.loads(d["unverified_claims"] or "[]")
    d["claims"] = json.loads(d.get("claims") or "[]")
    d["source_titles"] = json.loads(d.get("source_titles") or "{}")
    d["tokens_in"] = int(d["tokens_in"] or 0)
    d["tokens_out"] = int(d["tokens_out"] or 0)
    return d


def list_tasks(goal_id: str | None = None) -> list[dict]:
    with _connect() as conn:
        if goal_id:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE goal_id=? ORDER BY created_at", (goal_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tasks ORDER BY created_at").fetchall()
    return [_row_to_task_dict(r) for r in rows]


def get_task(task_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _row_to_task_dict(row) if row else None


def save_task(task: Task) -> None:
    d = task.model_dump(mode="json")
    with _connect() as conn:
        conn.execute(
            """UPDATE tasks SET title=?, description=?, status=?, assigned_agent=?,
                evidence=?, evidence_texts=?, unverified_claims=?, result=?,
                retries=?, max_retries=?, tokens_in=?, tokens_out=?,
                claims=?, source_titles=?, updated_at=? WHERE id=?""",
            (
                d["title"], d["description"], d["status"], d["assigned_agent"],
                json.dumps(d["evidence"]), json.dumps(d["evidence_texts"]),
                json.dumps(d["unverified_claims"]), d["result"], d["retries"],
                d["max_retries"], d["tokens_in"], d["tokens_out"],
                json.dumps(d["claims"]), json.dumps(d["source_titles"]),
                d["updated_at"], d["id"],
            ),
        )


def claim_task(task_id: str) -> dict | None:
    """Atomically claim a PENDING/NEEDS_RETRY task for background execution.

    The UPDATE ... WHERE status IN (...) is a single statement, so two
    concurrent workers can't both claim the same task: the loser gets
    rowcount 0 and this returns None. Returns the task dict if claimed.
    """
    with _connect() as conn:
        cur = conn.execute(
            """UPDATE tasks SET status='in_progress', updated_at=?
               WHERE id=? AND status IN ('pending','needs_retry')""",
            (_now(), task_id),
        )
        if cur.rowcount == 0:
            return None
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _row_to_task_dict(row) if row else None

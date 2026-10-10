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
                result TEXT,
                retries INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 2,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(goal_id) REFERENCES goals(id))"""
        )


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
                evidence, evidence_texts, result, retries, max_retries,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                d["id"], goal_id, d["title"], d["description"], d["status"],
                d["assigned_agent"], json.dumps(d["evidence"]),
                json.dumps(d["evidence_texts"]), d["result"], d["retries"],
                d["max_retries"], d["created_at"], d["updated_at"],
            ),
        )


def _row_to_task_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["evidence"] = json.loads(d["evidence"] or "[]")
    d["evidence_texts"] = json.loads(d["evidence_texts"] or "[]")
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
                evidence=?, evidence_texts=?, result=?, retries=?, max_retries=?,
                updated_at=? WHERE id=?""",
            (
                d["title"], d["description"], d["status"], d["assigned_agent"],
                json.dumps(d["evidence"]), json.dumps(d["evidence_texts"]),
                d["result"], d["retries"], d["max_retries"], d["updated_at"], d["id"],
            ),
        )

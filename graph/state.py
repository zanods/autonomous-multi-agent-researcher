from __future__ import annotations

from typing import TypedDict

from models.task import Task


class AgentState(TypedDict, total=False):
    """Shared state threaded through the LangGraph workflow."""

    query: str
    tasks: list[Task]
    current_task_id: str | None
    findings: list[dict]
    report: str | None
    run_log: list[str]
    mock_mode: bool

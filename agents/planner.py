from __future__ import annotations

from pydantic import BaseModel, Field

from agents.llm import get_llm
from models.task import Task


class TaskDraft(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = ""


class TaskPlan(BaseModel):
    tasks: list[TaskDraft] = Field(min_length=1, max_length=8)


_MOCK_SUBTASKS = [
    ("Background & definitions", "Gather key definitions and background context."),
    ("Current evidence & data", "Collect recent data points and sources."),
    ("Risks, trade-offs & outlook", "Identify risks, trade-offs and future outlook."),
]


def plan_tasks(query: str, mock_mode: bool = True) -> list[Task]:
    llm = None if mock_mode else get_llm()
    if mock_mode or llm is None:
        short = query[:60]
        return [
            Task(title=f"{title} — {short}", description=desc, assigned_agent="researcher")
            for title, desc in _MOCK_SUBTASKS
        ]
    try:
        plan: TaskPlan = llm.with_structured_output(TaskPlan).invoke(
            "Decompose this research query into 3-5 concrete subtasks. "
            "Each subtask needs a short title and a one-line description. "
            f"Query: {query}"
        )
    except Exception:
        # Free-tier models often fumble tool calling — fall back to raw JSON.
        plan = _plan_via_json(llm, query)
    return [
        Task(title=d.title, description=d.description, assigned_agent="researcher")
        for d in plan.tasks
    ]


def _plan_via_json(llm, query: str) -> TaskPlan:
    """Fallback planner: ask for raw JSON and parse it manually."""
    import json
    import re

    out = llm.invoke(
        "Decompose this research query into 3-5 concrete subtasks. "
        'Reply with ONLY a JSON object, no other text, like: '
        '{"tasks": [{"title": "Subtask title", "description": "one-line description"}]}. '
        f"Query: {query}"
    )
    text = out.content if hasattr(out, "content") else str(out)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {"tasks": []}
        return TaskPlan(**data)
    except Exception:
        short = query[:60]
        return TaskPlan(
            tasks=[
                TaskDraft(title=f"{title} — {short}", description=desc)
                for title, desc in _MOCK_SUBTASKS
            ]
        )

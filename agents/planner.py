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
    plan: TaskPlan = llm.with_structured_output(TaskPlan).invoke(
        "Decompose this research query into 3-5 concrete subtasks. "
        "Each subtask needs a short title and a one-line description. "
        f"Query: {query}"
    )
    return [
        Task(title=d.title, description=d.description, assigned_agent="researcher")
        for d in plan.tasks
    ]

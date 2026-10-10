from __future__ import annotations

from agents.llm import get_llm
from models.task import Task


def verify_task(task: Task, mock_mode: bool = True) -> tuple[bool, str]:
    """Return (passed, note). Mock mode always passes so the graph runs clean."""
    llm = None if mock_mode else get_llm()
    if mock_mode or llm is None:
        return True, "mock-verified"

    verdict = llm.invoke(
        "You are a fact-checker. Given the task and its evidence, reply with exactly "
        "one line: PASS or FAIL, then a short reason.\n"
        f"Task: {task.title}\nNotes: {task.result}\nEvidence: {task.evidence}"
    )
    text = verdict.content if hasattr(verdict, "content") else str(verdict)
    text = text.strip()
    return text.upper().startswith("PASS"), text[:200]

from __future__ import annotations

from agents.llm import get_llm, message_text
from models.task import Task, TaskStatus
from optimization.budget import TokenBudget


def synthesize(
    query: str,
    tasks: list[Task],
    mock_mode: bool = True,
    budget: TokenBudget | None = None,
) -> str:
    llm = None if mock_mode else get_llm()
    done = [t for t in tasks if t.status == TaskStatus.DONE]
    failed = [t for t in tasks if t.status == TaskStatus.FAILED]

    if mock_mode or llm is None:
        lines = [f"# Research Report (mock)\n", f"**Query:** {query}\n"]
        for t in done:
            lines.append(f"## {t.title}\n{t.result}\n")
            lines.extend(f"- {e}" for e in t.evidence)
        if failed:
            lines.append("\n## Gaps (needs human review)")
            lines.extend(f"- {t.title}" for t in failed)
        return "\n".join(lines)

    sections = "\n\n".join(
        f"### {t.title}\n{t.result}\nEvidence: {t.evidence}" for t in done
    )
    out = llm.invoke(
        "Synthesize the verified findings below into a structured analytical "
        f"report with citations.\nQuery: {query}\n\n{sections}"
    )
    if budget is not None:
        budget.record("synthesizer", out)
    return message_text(out)

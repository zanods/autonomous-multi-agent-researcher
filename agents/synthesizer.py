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
    done = [t for t in tasks if t.status == TaskStatus.DONE]
    failed = [t for t in tasks if t.status == TaskStatus.FAILED]

    if not done:
        # Nothing verified: do not let the LLM write a report from its own
        # knowledge. Say so plainly instead.
        text = (
            "# Research Report\n\n"
            f"**Query:** {query}\n\n"
            "No research tasks could be verified after retries, so this "
            "report contains no verified findings.\n"
        )
        if failed:
            text += (
                "\n## Verification gaps\n"
                "The following research tasks could not be verified after "
                "retries and were excluded from this report:\n"
            )
            text += "\n".join(f"- {t.title}" for t in failed)
        return text

    llm = None if mock_mode else get_llm()

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
    missing = "\n".join(f"- {t.title}" for t in failed)
    out = llm.invoke(
        "Synthesize ONLY the verified findings below into a structured "
        "analytical report with citations.\n"
        "STRICT RULES:\n"
        "- Use no information beyond what is provided below. Do NOT add "
        "facts, figures, or conclusions from your own knowledge.\n"
        "- Do NOT write about the topics listed under 'Missing evidence' — "
        "no verified findings exist for them.\n"
        "- If part of the query cannot be answered from the verified "
        "findings, say so explicitly instead of filling the gap.\n"
        f"Query: {query}\n\n"
        f"Verified findings:\n{sections}\n\n"
        f"Missing evidence (do not cover these topics):\n{missing if missing else '(none)'}"
    )
    if budget is not None:
        budget.record("synthesizer", out)
    text = message_text(out)
    if failed:
        # Never let a confident report hide that some tasks failed
        # verification — the reader must know what was excluded.
        text += (
            "\n\n## Verification gaps\n"
            "The following research tasks could not be verified after retries "
            "and were excluded from this report:\n"
        )
        text += "\n".join(f"- {t.title}" for t in failed)
    return text

from __future__ import annotations

from agents.llm import get_llm, message_text
from models.task import Claim, Task, TaskStatus
from optimization.budget import TokenBudget


def build_citation_map(tasks: list[Task]) -> tuple[dict[str, int], dict[str, str]]:
    """Number each cited source ref in order of first appearance.

    Returns (ref -> n, ref -> display title). Only verified claims with a
    source are cited; a claim without a source never gets a number.
    """
    numbers: dict[str, int] = {}
    titles: dict[str, str] = {}
    for t in tasks:
        for c in t.claims:
            if c.verified and c.source and c.source not in numbers:
                numbers[c.source] = len(numbers) + 1
                titles[c.source] = t.source_titles.get(c.source, c.source)
    return numbers, titles


def _sources_section(numbers: dict[str, int], titles: dict[str, str]) -> str:
    lines = ["## Sources"]
    for ref, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        lines.append(f"[{n}] {titles[ref]} — {ref}")
    return "\n".join(lines)


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
        numbers, titles = build_citation_map(done)
        for t in done:
            lines.append(f"## {t.title}\n")
            if t.claims:
                for c in t.claims:
                    if not c.verified:
                        continue
                    tag = f" [{numbers[c.source]}]" if c.source in numbers else ""
                    lines.append(f"- {c.text}{tag}")
            else:
                lines.append(f"{t.result}\n")
                lines.extend(f"- {e}" for e in t.evidence)
        if numbers:
            lines.append("\n" + _sources_section(numbers, titles))
        if failed:
            lines.append("\n## Gaps (needs human review)")
            lines.extend(f"- {t.title}" for t in failed)
        return "\n".join(lines)

    numbers, titles = build_citation_map(done)
    sections = []
    for t in done:
        parts = [f"### {t.title}"]
        if t.claims:
            for c in t.claims:
                if not c.verified:
                    continue
                tag = f"[{numbers[c.source]}] " if c.source in numbers else ""
                parts.append(f"- {tag}{c.text}")
        else:
            parts.append(t.result or "")
        sections.append("\n".join(parts))
    sections_text = "\n\n".join(sections)
    sources_text = _sources_section(numbers, titles) if numbers else "(no cited sources)"
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
        "CITATION RULES:\n"
        "- Every factual statement you write must end with its citation "
        "number exactly as given in the findings (e.g. [1], [2]). Do NOT "
        "renumber, and do NOT invent citation numbers.\n"
        "- Findings without a citation number have no verified source: "
        "state them without any citation.\n"
        "- End the report with a '## Sources' section listing every "
        "citation number with its title and URL, in numeric order.\n"
        f"Query: {query}\n\n"
        f"Verified findings (with citation numbers):\n{sections_text}\n\n"
        f"Sources (use these titles/URLs in the '## Sources' section):\n{sources_text}\n\n"
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

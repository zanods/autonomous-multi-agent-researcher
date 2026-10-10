from __future__ import annotations

import os

from agents.llm import get_llm, message_text
from models.task import Task
from optimization.budget import TokenBudget
from optimization.claims import exact_match_verify, split_claims

OPTIMIZE = os.getenv("OPTIMIZE_TOKENS", "1") == "1"


def verify_task(
    task: Task,
    mock_mode: bool = True,
    optimize: bool = OPTIMIZE,
    budget: TokenBudget | None = None,
) -> tuple[bool, str]:
    """Two-stage fact check when optimize=True:

    1. exact keyword match per claim (0 LLM tokens),
    2. LLM judges only the claims that failed stage 1, with trimmed evidence.

    Falls back to the classic whole-task prompt when optimize=False.
    """
    if mock_mode:
        return True, "mock-verified"
    llm = get_llm()
    if llm is None:
        return True, "mock-verified (no LLM configured)"
    if optimize:
        return _verify_staged(task, llm, budget)
    return _verify_whole(task, llm, budget)


def _verify_whole(
    task: Task, llm, budget: TokenBudget | None = None
) -> tuple[bool, str]:
    verdict = llm.invoke(
        "You are a fact-checker. Given the task and its evidence, reply with exactly "
        "one line: PASS or FAIL, then a short reason.\n"
        f"Task: {task.title}\nNotes: {task.result}\nEvidence: {task.evidence}"
    )
    if budget is not None:
        budget.record("fact_checker", verdict)
    text = message_text(verdict).strip()
    return text.upper().startswith("PASS"), text[:200]


def _verify_staged(
    task: Task, llm, budget: TokenBudget | None = None
) -> tuple[bool, str]:
    claims = split_claims(task.result or "")
    if not claims:
        return False, "staged: no verifiable claims extracted"
    texts = task.evidence_texts or []

    auto_ok, need_llm = [], []
    for claim in claims:
        (auto_ok if exact_match_verify(claim, texts) else need_llm).append(claim)

    llm_ok, llm_bad = [], []
    if need_llm:
        snippets = "\n".join(f"- {t[:600]}" for t in texts[:3])
        if budget is not None:
            snippets = budget.spend("fact_checker", snippets)
        prompt = (
            "Fact-check each numbered claim against the evidence snippets below. "
            "Reply with exactly one line per claim: '<n>. PASS' or '<n>. FAIL'.\n"
            + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(need_llm))
            + f"\nEvidence:\n{snippets}"
        )
        out = llm.invoke(prompt)
        if budget is not None:
            budget.record("fact_checker", out)
        text = message_text(out)
        for i, line in enumerate(text.strip().splitlines()):
            target = need_llm[i] if i < len(need_llm) else line
            (llm_ok if "PASS" in line.upper() else llm_bad).append(target)

    total_ok = len(auto_ok) + len(llm_ok)
    note = (
        f"staged: {total_ok}/{len(claims)} claims verified "
        f"({len(auto_ok)} exact-match, {len(llm_ok)} LLM-verified, {len(llm_bad)} failed)"
    )
    # Every claim must be verified — an empty/garbled LLM reply verifies
    # nothing, so it must not pass vacuously.
    return total_ok == len(claims), note

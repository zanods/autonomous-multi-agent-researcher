from __future__ import annotations

import os
import re

from agents.llm import get_llm, message_text
from models.task import Claim, Task
from optimization.budget import TokenBudget
from optimization.claims import _match_chunk, exact_match_verify, split_claims

# Re-exported for backward compatibility (tests patch fc_mod.exact_match_verify).
__all__ = ["verify_task", "exact_match_verify"]

OPTIMIZE = os.getenv("OPTIMIZE_TOKENS", "1") == "1"

# Strict verdict format: "1. PASS" / "2: FAIL". Anything else is fail-closed —
# a loose `"PASS" in line` check would let "2: FAIL - does not PASS" through.
_VERDICT = re.compile(r"^\s*\d+\s*[:.]\s*(PASS|FAIL)\b", re.IGNORECASE)


def verify_task(
    task: Task,
    mock_mode: bool = True,
    optimize: bool = OPTIMIZE,
    budget: TokenBudget | None = None,
) -> tuple[bool, str, list[str]]:
    """Two-stage fact check when optimize=True:

    1. exact keyword match per claim (0 LLM tokens),
    2. LLM judges only the claims that failed stage 1, with trimmed evidence.

    Falls back to the classic whole-task prompt when optimize=False.

    Returns (ok, note, failed_claims) — the failed claims let a retry focus
    its research instead of re-doing everything from scratch.
    """
    if mock_mode:
        return True, "mock-verified", []
    llm = get_llm()
    if llm is None:
        return True, "mock-verified (no LLM configured)", []
    if optimize:
        return _verify_staged(task, llm, budget)
    return _verify_whole(task, llm, budget)


def _verify_whole(
    task: Task, llm, budget: TokenBudget | None = None
) -> tuple[bool, str, list[str]]:
    verdict = llm.invoke(
        "You are a fact-checker. Given the task and its evidence, reply with exactly "
        "one line: PASS or FAIL, then a short reason.\n"
        f"Task: {task.title}\nNotes: {task.result}\nEvidence: {task.evidence}"
    )
    if budget is not None:
        budget.record("fact_checker", verdict)
    text = message_text(verdict).strip()
    return text.upper().startswith("PASS"), text[:200], []


def _verify_staged(
    task: Task, llm, budget: TokenBudget | None = None
) -> tuple[bool, str]:
    # Prefer researcher-attached claims (they carry source refs); fall back
    # to splitting raw notes for tasks built without the researcher.
    claims: list[Claim] = (
        task.claims
        if task.claims
        else [Claim(text=c) for c in split_claims(task.result or "")]
    )
    if not claims:
        return False, "staged: no verifiable claims extracted", []
    texts = task.evidence_texts or []
    refs = (
        task.evidence
        if len(task.evidence) == len(texts)
        else [None] * len(texts)
    )

    auto_ok, need_llm = [], []
    for claim in claims:
        idx = _match_chunk(claim.text, texts)
        if idx is not None:
            # Exact match: the matched chunk is the authoritative source —
            # prefer it over the researcher's looser attachment. "unknown"
            # is a display placeholder, never a citable source.
            if refs[idx] not in (None, "unknown"):
                claim.source = refs[idx]
            claim.verified = True
            auto_ok.append(claim)
        else:
            need_llm.append(claim)

    llm_ok, llm_bad = [], []
    if need_llm:
        snippets = "\n".join(f"- {t[:600]}" for t in texts[:3])
        if budget is not None:
            snippets = budget.spend("fact_checker", snippets)
        prompt = (
            "Fact-check each numbered claim against the evidence snippets below. "
            "Reply with exactly one line per claim: '<n>. PASS' or '<n>. FAIL'.\n"
            + "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(need_llm))
            + f"\nEvidence:\n{snippets}"
        )
        out = llm.invoke(prompt)
        if budget is not None:
            budget.record("fact_checker", out)
        text = message_text(out)
        for i, line in enumerate(text.strip().splitlines()):
            target = need_llm[i] if i < len(need_llm) else None
            m = _VERDICT.match(line)
            if target is None:
                continue
            if m and m.group(1).upper() == "PASS":
                target.verified = True  # keep researcher's attached source
                llm_ok.append(target)
            else:
                llm_bad.append(target)

    total_ok = len(auto_ok) + len(llm_ok)
    note = (
        f"staged: {total_ok}/{len(claims)} claims verified "
        f"({len(auto_ok)} exact-match, {len(llm_ok)} LLM-verified, {len(llm_bad)} failed)"
    )
    # Failed = explicit FAILs + claims the LLM never judged (short/garbled
    # reply). Identity-based: the lists hold the same objects as `claims`.
    judged = set(map(id, llm_ok)) | set(map(id, llm_bad))
    failed = list(llm_bad) + [c for c in need_llm if id(c) not in judged]
    # Every claim must be verified — an empty/garbled LLM reply verifies
    # nothing, so it must not pass vacuously.
    task.claims = claims
    return (
        total_ok == len(claims),
        note,
        [c.text for c in failed],
    )

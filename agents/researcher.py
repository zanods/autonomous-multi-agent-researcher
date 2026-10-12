from __future__ import annotations

import os

from agents.llm import get_llm, message_text
from models.task import Task
from optimization.budget import TokenBudget
from optimization.claims import attach_claim_sources, split_claims
from rag.retriever import rag_search
from tools.web_search import web_search

OPTIMIZE = os.getenv("OPTIMIZE_TOKENS", "1") == "1"


def research_task(
    task: Task,
    mock_mode: bool = True,
    optimize: bool = OPTIMIZE,
    budget: TokenBudget | None = None,
    focus_claims: list[str] | None = None,
) -> Task:
    """Research a task. When focus_claims is given (a retry after failed
    verification), research narrows to those claims instead of broadening
    the topic — retries must shrink the problem, not grow it."""
    llm = None if mock_mode else get_llm()
    if mock_mode or llm is None:
        task.evidence = [
            f"[mock] Source A — key finding relevant to '{task.title}'.",
            f"[mock] Source B — supporting data point for '{task.title}'.",
        ]
        task.result = f"[mock] Research notes for: {task.title}\n{task.description}"
        return task

    query = f"{task.title} {task.description}".strip()

    # 1. Internal knowledge base (ChromaDB) — [] when nothing ingested yet.
    rag_hits = rag_search(query, k=4)
    # 2. Live web search (Tavily) — [] when no API key.
    web_hits = web_search(query, max_results=5)

    rag_texts = [h["text"] for h in rag_hits]
    web_texts = [f"{h['title']}: {h['snippet']}" for h in web_hits]
    new_texts = rag_texts + web_texts
    if focus_claims:
        # Retry: accumulate evidence, don't discard what was already found.
        seen = set(task.evidence_texts)
        task.evidence_texts = task.evidence_texts + [t for t in new_texts if t not in seen]
    else:
        task.evidence_texts = new_texts

    context_parts = [f"[KB {h['id']}] {t[:800]}" for h, t in zip(rag_hits, rag_texts)]
    context_parts += [f"[WEB] {t}" for t in web_texts]
    context = "\n".join(context_parts) or "(no retrieval results; using model knowledge)"

    if optimize:
        budget = budget or TokenBudget()
        context = budget.spend("researcher", context)

    if focus_claims:
        prompt = (
            "Earlier research left the following claims UNVERIFIED for lack of "
            "evidence. Find targeted evidence for exactly these claims — do not "
            "broaden the topic. Write SHORT notes: one paragraph per claim, "
            "covering only the claims below.\n"
            + "\n".join(f"- {c}" for c in focus_claims)
            + f"\nTask: {task.title}\nContext:\n{context}"
        )
    else:
        prompt = (
            "Write concise research notes for the task below, grounded in the context. "
            "Cite sources like [KB ...] or [WEB] where used.\n"
            f"Task: {task.title}\n{task.description}\nContext:\n{context}"
        )
    notes = llm.invoke(prompt)
    if budget is not None:
        budget.record("researcher", notes)
    task.result = message_text(notes)
    # One ordered evidence list: (ref, title, text) with ref a URL,
    # "kb:<chunk_id>", or None when unknown. task.evidence and
    # task.evidence_texts stay parallel to it so the fact-checker can map a
    # matched chunk back to its ref.
    new_items: list[tuple[str | None, str, str]] = [
        (f"kb:{h['id']}", str(h["id"]), t) for h, t in zip(rag_hits, rag_texts)
    ]
    new_items += [
        (h.get("url") or None, h.get("title") or "web result", t)
        for h, t in zip(web_hits, web_texts)
    ]
    if focus_claims:
        # Retry: accumulate evidence items, don't discard what was already found.
        old_texts = task.evidence_texts or []
        old_refs = (
            task.evidence
            if len(task.evidence) == len(old_texts)
            else [None] * len(old_texts)
        )
        old_titles = task.source_titles or {}
        items: list[tuple[str | None, str, str]] = [
            (
                r if r and r != "unknown" else None,
                old_titles.get(r, r or "unknown") if r else "unknown",
                t,
            )
            for r, t in zip(old_refs, old_texts)
        ]
        seen = set(old_texts)
        items += [it for it in new_items if it[2] not in seen]
    else:
        items = new_items
    task.evidence = [ref or "unknown" for ref, _, _ in items]
    task.evidence_texts = [t for _, _, t in items]
    task.source_titles = {ref: title for ref, title, _ in items if ref}
    task.claims = attach_claim_sources(split_claims(task.result or ""), items)
    return task

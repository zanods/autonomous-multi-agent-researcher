from __future__ import annotations

from agents.llm import get_llm
from models.task import Task
from rag.retriever import rag_search
from tools.web_search import web_search


def research_task(task: Task, mock_mode: bool = True) -> Task:
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

    context_parts = [f"[KB {h['id']}] {h['text'][:800]}" for h in rag_hits]
    context_parts += [f"[WEB] {h['title']}: {h['snippet']}" for h in web_hits]
    context = "\n".join(context_parts) or "(no retrieval results; using model knowledge)"

    notes = llm.invoke(
        "Write concise research notes for the task below, grounded in the context. "
        "Cite sources like [KB ...] or [WEB] where used.\n"
        f"Task: {task.title}\n{task.description}\nContext:\n{context}"
    )
    task.result = notes.content if hasattr(notes, "content") else str(notes)
    task.evidence = [h["url"] for h in web_hits if h.get("url")]
    task.evidence += [f"kb:{h['id']}" for h in rag_hits]
    return task

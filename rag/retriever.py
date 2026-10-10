from __future__ import annotations

import os

from rag.vectorstore import default_persist_dir, load_vectorstore


def rag_search(query: str, k: int = 4) -> list[dict]:
    """Search the local knowledge base. Returns [] when nothing ingested yet."""
    persist_dir = default_persist_dir()
    if not os.path.isdir(persist_dir):
        return []
    vs = load_vectorstore(persist_dir)
    docs = vs.similarity_search(query, k=k)
    hits = [
        {"id": d.metadata.get("source", f"doc-{i}"), "text": d.page_content}
        for i, d in enumerate(docs)
    ]
    return rerank(query, hits)


def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """Cohere Rerank when COHERE_API_KEY is set, else pass-through."""
    if not hits or not os.getenv("COHERE_API_KEY"):
        return hits
    import cohere

    client = cohere.Client(os.getenv("COHERE_API_KEY"))
    res = client.rerank(
        query=query,
        documents=[h["text"] for h in hits],
        top_n=min(len(hits), top_n),
        model="rerank-english-v3.0",
    )
    return [hits[r.index] for r in res.results]

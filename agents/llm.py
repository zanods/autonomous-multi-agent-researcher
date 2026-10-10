from __future__ import annotations

import os


def get_llm():
    """Return a ChatOpenAI instance, or None when no API key is configured.

    Agents fall back to deterministic mock outputs when this returns None,
    so the graph can be exercised end-to-end without spending API credits.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.2)

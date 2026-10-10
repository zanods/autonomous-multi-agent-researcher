from __future__ import annotations

import os


def _active_provider() -> str | None:
    # Gemini first: it has a free tier, good enough for testing live mode.
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def has_api_key() -> bool:
    return _active_provider() is not None


def get_llm():
    """Chat model for agents, or None when no API key is configured.

    Agents fall back to deterministic mock outputs when this returns None,
    so the graph can be exercised end-to-end without spending API credits.
    """
    provider = _active_provider()
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("MODEL", "gemini-2.0-flash"), temperature=0.2
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.2)
    return None
